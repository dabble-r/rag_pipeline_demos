"""
EXTRACT ALL TEXT + RETRIEVAL/RERANK (TEXT ONLY — IMAGES DROPPED)
- Parse stage: extract ALL text from PDF (rawdict), no relevancy discard
- One chunk per page; chunk_text splits for embedding; relevance at retrieval/rerank
- Query expansion, similarity threshold, cross-encoder reranking
"""

import os
import re
import numpy as np
from dotenv import load_dotenv
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from openai import OpenAI
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# EXTRACT ALL TEXT (NO RELEVANCY FILTER AT PARSE)
# ---------------------------------------------------------------------------

def parse_pdf_to_chunks(pdf_path, user_query):
    """
    Extracts ALL text from the PDF via rawdict. No discard based on relevancy;
    relevance is determined at retrieval/reranking stage.
    - One chunk per page (all span text in reading order)
    - Text layer only (no image processing)
    - user_query accepted for API compatibility but not used for filtering
    """

    doc = fitz.open(pdf_path)
    chunks = []

    for page_idx, page in enumerate(doc):
        raw = page.get_text("rawdict")
        spans = []

        for block in raw["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = (span.get("text") or "").strip()
                    if not text:
                        continue
                    bbox = span.get("bbox") or (0, 0, 0, 0)
                    spans.append({"text": text, "bbox": bbox, "y": bbox[1]})

        spans.sort(key=lambda s: s["y"])
        page_text = "\n".join(sp["text"] for sp in spans)

        if page_text.strip():
            chunks.append({
                "text": page_text,
                "metadata": {"page": page_idx + 1, "type": "section", "has_image": False},
            })

    return chunks


# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------

def chunk_text(text, page_number):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0,
    )
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0,
        tokens_per_chunk=256,
    )

    chunks = []
    for ch in character_splitter.split_text(text):
        for tok in token_splitter.split_text(ch):
            chunks.append({
                "text": tok,
                "metadata": {"page": page_number, "type": "section", "has_image": False}
            })
    return chunks


# ---------------------------------------------------------------------------
# QUERY EXPANSION
# ---------------------------------------------------------------------------

def _normalize_expansion_line(line: str) -> str:
    """Strip leading list numbering (e.g. '1. ', '2) ') for consistent query text."""
    return re.sub(r"^\s*\d+[.)]\s*", "", line.strip()).strip()


def generate_context_aware_expansions(user_query, retrieved_docs, model="gpt-3.5-turbo"):
    context = "\n\n".join(retrieved_docs[:5])
    prompt = f"""
You are a ceramics and glaze chemistry assistant.

The user asked:
"{user_query}"

Below are excerpts retrieved from the document:
---
{context}
---

Generate exactly 8 to 12 grounded follow-up questions.
- One question per line, no numbering or bullets.
- Each question must be concise, single-topic, and grounded only in the excerpts.
"""

    messages = [
        {"role": "system", "content": "You generate grounded ceramics/glaze queries. Output one question per line, no numbers."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    lines = response.choices[0].message.content.strip().split("\n")
    normalized = [_normalize_expansion_line(q) for q in lines if q.strip()]
    return [q for q in normalized if q]


def filter_expansions_by_relevance(original_query, expansions, cross_encoder):
    if not expansions:
        return []
    pairs = [[original_query, q] for q in expansions]
    scores = cross_encoder.predict(pairs)
    top_idx = np.argsort(scores)[::-1][:5]
    return [expansions[i] for i in top_idx]


# ---------------------------------------------------------------------------
# FINAL ANSWER
# ---------------------------------------------------------------------------

def generate_final_answer(original_query, context, model="gpt-3.5-turbo"):
    prompt = f"""
You are a ceramics artist. Answer the user's question using ONLY the following excerpts.
If the information is not in the excerpts, say so clearly.

Excerpts:
{context}

User question: {original_query}

Instructions:
- For recipe or ingredient questions: include full details from the excerpts (e.g. ingredients and amounts, cone, firing) when present; do not summarize to a single phrase if the excerpts contain a full recipe.
- Otherwise provide a concise answer.
"""
    messages = [
        {"role": "system", "content": "You answer using only provided excerpts. For recipe questions include full details when present."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# BUILD INDEX
# ---------------------------------------------------------------------------

PDF_PATH = "data/recipes.pdf"
COLLECTION_NAME = "glaze-collection"
ORIGINAL_QUERY = "What is the recipe for temmoku glaze?"

raw_groups = parse_pdf_to_chunks(PDF_PATH, ORIGINAL_QUERY)

docs = []
metas = []
for g in raw_groups:
    page = g["metadata"]["page"]
    for ch in chunk_text(g["text"], page):
        docs.append(ch["text"])
        metas.append(ch["metadata"])

ids = [str(i) for i in range(len(docs))]

if not docs:
    # Why this happens (e.g. PDF with images, text-only parsing): see filtered_query.md §2 and §6.
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
# RETRIEVAL → EXPANSION → RERANKING → ANSWER
# ---------------------------------------------------------------------------

TOP_K_RERANK = 7
N_RESULTS_PER_QUERY = 10
MIN_SIM_THRESHOLD = 0.45

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

initial = collection.query(
    query_texts=[ORIGINAL_QUERY],
    n_results=8,
    include=["documents", "metadatas", "distances"],
)
retrieved_for_expansion = initial["documents"][0]

raw_expansions = generate_context_aware_expansions(ORIGINAL_QUERY, retrieved_for_expansion)
aug_queries = filter_expansions_by_relevance(ORIGINAL_QUERY, raw_expansions, cross_encoder)

joint_queries = [ORIGINAL_QUERY] + aug_queries
results = collection.query(
    query_texts=joint_queries,
    n_results=N_RESULTS_PER_QUERY,
    include=["documents", "metadatas", "distances"],
)

unique = {}
for docs_list, metas_list, dist_list in zip(
    results["documents"], results["metadatas"], results["distances"]
):
    for d, m, dist in zip(docs_list, metas_list, dist_list):
        if d not in unique:
            unique[d] = {"meta": m, "dist": dist}

filtered_docs = [d for d, info in unique.items() if info["dist"] < MIN_SIM_THRESHOLD]
filtered_metas = [unique[d]["meta"] for d in filtered_docs]

if not filtered_docs:
    filtered_docs = list(unique.keys())
    filtered_metas = [unique[d]["meta"] for d in filtered_docs]

pairs = [[ORIGINAL_QUERY, d] for d in filtered_docs]
scores = cross_encoder.predict(pairs)

top_idx = np.argsort(scores)[::-1][:TOP_K_RERANK]
top_docs = [filtered_docs[i] for i in top_idx]

context = "\n\n".join(top_docs)
final_answer = generate_final_answer(ORIGINAL_QUERY, context)

print("Augmented queries:")
for q in aug_queries:
    print(" -", q)

print("\nFinal answer:")
print(final_answer)