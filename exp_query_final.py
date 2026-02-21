"""
Full pipeline from aug_result.md: context-aware query expansion, retrieval,
reranking with cross-encoder, and one final LLM response.
"""

import os

import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import CrossEncoder

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# LLM: context-aware query expansion
# ---------------------------------------------------------------------------


def generate_context_aware_queries(
    user_query: str,
    retrieved_docs: list[str],
    model: str = "gpt-3.5-turbo",
) -> list[str]:
    """
    Generate grounded, context-aware query expansions from the user question
    and the initially retrieved documents.
    """
    context = "\n\n".join(retrieved_docs[:3])
    prompt = f"""
You are a ceramics artist assistant working with ceramic glaze composition data.

The user asked:
"{user_query}"

Below are excerpts retrieved from the document:
---
{context}
---

Based on BOTH the user question and the retrieved content,
generate up to five follow-up questions that:
- explore different angles of the same topic,
- stay grounded in the retrieved content,
- avoid speculation,
- are concise and single-topic,
- help retrieve more relevant information from the document.

List each question on a separate line without numbering.
"""
    messages = [
        {"role": "system", "content": "You generate grounded financial queries."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content.strip().split("\n")
    return [q.strip() for q in content if q.strip()]


# ---------------------------------------------------------------------------
# LLM: one final answer from context
# ---------------------------------------------------------------------------


def generate_final_answer(
    original_query: str,
    context: str,
    model: str = "gpt-3.5-turbo",
) -> str:
    """
    Produce a single final answer to the user question using only the
    provided context (reranked document excerpts). 
    """
    prompt = f"""You are a ceramics artist. Answer the user's question using ONLY the following excerpts from a ceramics glaze composition report. If the information is not in the excerpts, say so clearly.

Excerpts:
{context}

User question: {original_query}

Provide one concise, direct answer. Do not speculate beyond the excerpts."""
    messages = [
        {
            "role": "system",
            "content": "You answer questions based only on the provided document excerpts.",
        },
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Document loading, splitting, Chroma
# ---------------------------------------------------------------------------

PDF_PATH = "data/recipes.pdf"
COLLECTION_NAME = "microsoft-collection"
INITIAL_QUERY = "What are the ingredients in a typical ceramic glaze composition?"
ORIGINAL_QUERY = (
    "What is the recipe for temmoku glaze?"
)
TOP_K_RERANK = 5
N_RESULTS_PER_QUERY = 10

reader = PdfReader(PDF_PATH)
pdf_texts = [p.extract_text().strip() for p in reader.pages]
pdf_texts = [t for t in pdf_texts if t]

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0,
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0,
    tokens_per_chunk=256,
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME,
    embedding_function=embedding_function,
)
if chroma_collection.count() == 0:
    ids = [str(i) for i in range(len(token_split_texts))]
    chroma_collection.add(ids=ids, documents=token_split_texts)

# ---------------------------------------------------------------------------
# Step 1: Initial retrieval → context-aware augmented queries
# ---------------------------------------------------------------------------

results_initial = chroma_collection.query(
    query_texts=[INITIAL_QUERY],
    n_results=5,
    include=["documents"],
)
retrieved_for_expansion = results_initial["documents"][0]
aug_queries = generate_context_aware_queries(
    INITIAL_QUERY,
    retrieved_for_expansion,
)

# ---------------------------------------------------------------------------
# Step 2: Joint retrieval (original + aug queries) → deduplicate
# ---------------------------------------------------------------------------

joint_query = [ORIGINAL_QUERY] + aug_queries
results = chroma_collection.query(
    query_texts=joint_query,
    n_results=N_RESULTS_PER_QUERY,
    include=["documents"],
)
retrieved_documents = results["documents"]

unique_documents = set()
for doc_list in retrieved_documents:
    for doc in doc_list:
        unique_documents.add(doc)
unique_doc_list = list(unique_documents)

# ---------------------------------------------------------------------------
# Step 3: Rerank by relevance to original query → top_k documents
# ---------------------------------------------------------------------------

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[ORIGINAL_QUERY, doc] for doc in unique_doc_list]
scores = cross_encoder.predict(pairs)
top_indices = np.argsort(scores)[::-1][:TOP_K_RERANK]
top_documents = [unique_doc_list[i] for i in top_indices]

# ---------------------------------------------------------------------------
# Step 4: Single final answer from reranked context
# ---------------------------------------------------------------------------

context = "\n\n".join(top_documents)
final_answer = generate_final_answer(ORIGINAL_QUERY, context)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

print("Augmented queries:")
for q in aug_queries:
    print(f"  - {q}")
print()
print("Final answer:")
print(final_answer)
