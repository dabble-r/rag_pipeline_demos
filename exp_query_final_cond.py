"""
EXTRACT ALL TEXT: text layer + OCR from images (with OCR cleaning).
- Parse: rawdict per page + image OCR when text layer is missing/minimal
- No relevancy filter at parse; relevance at retrieval/reranking
- Query expansion (conditional), cross-encoder reranking
"""

import os
import io
import re
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# MODEL ROUTER
# ---------------------------------------------------------------------------

def choose_model(task: str) -> str:
    task = task.lower()
    if task == "classify":
        return "gpt-4o-mini"
    if task == "expand":
        return "gpt-4o-mini"
    if task == "answer":
        return "gpt-4o"
    return "gpt-4o-mini"


# ---------------------------------------------------------------------------
# CLOSED vs OPEN CLASSIFIER
# ---------------------------------------------------------------------------

def classify_query_type(user_query):
    model = choose_model("classify")

    prompt = f"""
Classify the user's question as either CLOSED or OPEN.

CLOSED questions:
- ask for a specific fact, recipe, number, ingredient list, or discrete detail
- can be answered by a single chunk of text

OPEN questions:
- ask for explanations, factors, causes, comparisons, summaries, or conceptual info
- require multiple chunks or broader reasoning

User question:
"{user_query}"

Respond with only one word: CLOSED or OPEN.
"""

    messages = [
        {"role": "system", "content": "You classify questions."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    label = response.choices[0].message.content.strip().upper()
    return "open" if "OPEN" in label else "closed"


def should_expand_query(user_query, initial_docs, threshold=2):
    qtype = classify_query_type(user_query)
    if qtype == "open":
        return True
    if len(initial_docs) < threshold:
        return True
    return False


# ---------------------------------------------------------------------------
# OCR CLEANING (for image-derived text)
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> str:
    """Keep lines that look like glaze/recipe content; drop noise and short junk."""
    mineral_terms = [
        "feldspar", "whiting", "silica", "kaolin", "ball clay", "iron",
        "copper", "cobalt", "manganese", "rutile", "zircon", "tin",
    ]
    lines = text.splitlines()
    cleaned = []
    for raw in (l.strip() for l in lines):
        if not raw or len(raw) < 3:
            continue
        if sum(c.isalnum() for c in raw) / len(raw) < 0.5:
            continue
        if re.fullmatch(r"[-=~_.,:;]+", raw):
            continue
        if any(term in raw.lower() for term in mineral_terms):
            cleaned.append(raw)
            continue
        if re.search(r"\d|%|cone\s*\d", raw.lower()):
            cleaned.append(raw)
            continue
    if sum(len(l) for l in cleaned) < 10:
        return ""
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# IMPROVED RAWDICT PARSER (TEXT ONLY — IMAGES DROPPED)
# ---------------------------------------------------------------------------

def parse_pdf_to_chunks(pdf_path, user_query=None):
    pdf_document = pdf_path
    doc = fitz.open(pdf_document)

    chunks = []

    for page_idx in range(doc.page_count):
        raw = doc.get_page_text(page_idx)
        page_text = (raw or "").strip()
        if not page_text:
            continue

        chunks.append({
            "text": page_text,
            "metadata": {"page": page_idx + 1, "type": "section"},
        })

    doc.close()
    return chunks


# ---------------------------------------------------------------------------
# IMPROVED CHUNKING (NO OVER-FRAGMENTATION)
# ---------------------------------------------------------------------------

def chunk_text(text, page_number):
    return [{
        "text": text,
        "metadata": {"page": page_number, "type": "section"},
    }]

# ---------------------------------------------------------------------------
# QUERY EXPANSION
# ---------------------------------------------------------------------------

def generate_context_aware_queries(user_query, retrieved_docs):
    model = choose_model("expand")

    context = "\n\n".join(retrieved_docs[:3])
    prompt = f"""
You are a ceramics artist assistant working with glaze composition data.

The user asked:
"{user_query}"

Below are excerpts retrieved from the document:
---
{context}
---

Generate up to five grounded follow-up questions.
Each question must be concise, single-topic, and grounded in the excerpts.
"""

    messages = [
        {"role": "system", "content": "You generate grounded glaze queries."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    lines = response.choices[0].message.content.strip().split("\n")
    return [q.strip() for q in lines if q.strip()]


# ---------------------------------------------------------------------------
# FINAL ANSWER
# ---------------------------------------------------------------------------

def generate_final_answer(original_query, context):
    model = choose_model("answer")

    prompt = f"""
You are a ceramics artist. Answer the user's question using ONLY the following excerpts.
If the information is not in the excerpts, say so clearly.

Excerpts:
{context}

User question: {original_query}

Provide one concise, direct answer.
"""

    messages = [
        {"role": "system", "content": "You answer using only provided excerpts."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# BUILD INDEX
# ---------------------------------------------------------------------------

PDF_PATH = "data/recipes.pdf"
COLLECTION_NAME = "glaze-collection"
ORIGINAL_QUERY = "What is the recipe for temmoku glaze?"
TOP_K_RERANK = 5
N_RESULTS_PER_QUERY = 10
MIN_SIM_THRESHOLD = 0.35

raw_groups = parse_pdf_to_chunks(PDF_PATH, ORIGINAL_QUERY)
# print("raw groups:", raw_groups)


docs = []
metas = []
for g in raw_groups:
    page = g["metadata"]["page"]
    # print("page: ", page)
    for ch in chunk_text(g["text"], page):
        docs.append(ch["text"])
        metas.append(ch["metadata"])

for i, d in enumerate(docs[:3]):
    print("CHUNK", i, "\n", d[:500], "\n---\n")

ids = [str(i) for i in range(len(docs))]

if not docs:
    raise SystemExit(
        "No documents to add: parse_pdf_to_chunks produced no chunks (empty PDF, no matching sections, or no text in spans)."
    )

embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME,
    embedding_function=embedding_function,
)

if collection.count() == 0:
    collection.add(ids=ids, documents=docs, metadatas=metas)


# ---------------------------------------------------------------------------
# RETRIEVAL → CONDITIONAL EXPANSION
# ---------------------------------------------------------------------------

initial = collection.query(
    query_texts=[ORIGINAL_QUERY],
    n_results=5,
    include=["documents"],
)
initial_docs = initial["documents"][0]

expand = should_expand_query(ORIGINAL_QUERY, initial_docs)

if expand:
    raw_expansions = generate_context_aware_queries(ORIGINAL_QUERY, initial_docs)
    aug_queries = raw_expansions
    joint_queries = [ORIGINAL_QUERY] + aug_queries
else:
    aug_queries = []
    joint_queries = [ORIGINAL_QUERY]

results = collection.query(
    query_texts=joint_queries,
    n_results=N_RESULTS_PER_QUERY,
    include=["documents"],
)

unique_docs = []
seen = set()
for doc_list in results["documents"]:
    for d in doc_list:
        if d not in seen:
            unique_docs.append(d)
            seen.add(d)

unique_doc_list = list(unique_docs)


# ---------------------------------------------------------------------------
# CROSS-ENCODER RERANKING
# ---------------------------------------------------------------------------

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[ORIGINAL_QUERY, doc] for doc in unique_doc_list]
scores = cross_encoder.predict(pairs)

top_indices = np.argsort(scores)[::-1][:TOP_K_RERANK]
top_documents = [unique_doc_list[i] for i in top_indices]


# ---------------------------------------------------------------------------
# FINAL ANSWER
# ---------------------------------------------------------------------------

context = "\n\n".join(top_documents)
final_answer = generate_final_answer(ORIGINAL_QUERY, context)

print("\nAugmented queries:")
for q in aug_queries:
    print(" -", q)

print("\nFinal answer:")
print(final_answer)