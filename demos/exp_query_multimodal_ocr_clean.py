"""
RAG PIPELINE WITH FIXES:
1) Use ORIGINAL_QUERY for initial retrieval + expansion
2) Fix system prompt domain for query expansion
3) Include/use metadatas, prefer section chunks for recipe-like questions
4) Basic OCR cleaning + downweight figure chunks in retrieval
"""

import os
import io
import re
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import fitz  # PyMuPDF
import pytesseract

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from openai import OpenAI
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# PDF → layout-aware chunks (figures + captions + OCR + section text)
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> str:
    """Basic OCR cleaning: remove very short/noisy lines."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # discard lines that are mostly non-alphanumeric or too short
        if len(line) < 4:
            continue
        if sum(c.isalnum() for c in line) / max(len(line), 1) < 0.4:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def find_nearest_caption(text_blocks, img_bbox):
    """Heuristic: caption is the closest text block vertically."""
    ix0, iy0, ix1, iy1 = img_bbox
    best = None
    best_dist = 999999
    for tb in text_blocks:
        tx0, ty0, tx1, ty1 = tb["bbox"]
        dist = min(abs(ty0 - iy1), abs(iy0 - ty1))
        if dist < best_dist:
            best_dist = dist
            best = tb
    return best

def collect_nearby_text(text_blocks, img_bbox, threshold=80):
    """Collect text blocks within a vertical window around the image."""
    ix0, iy0, ix1, iy1 = img_bbox
    collected = []
    for tb in text_blocks:
        tx0, ty0, tx1, ty1 = tb["bbox"]
        if abs(ty0 - iy1) < threshold or abs(iy0 - ty1) < threshold:
            collected.append(tb["text"])
    return "\n".join(collected)

def chunk_page_text(page_text, page_number):
    """Use existing splitters to chunk normal text."""
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
    for ch in character_splitter.split_text(page_text):
        for tok in token_splitter.split_text(ch):
            chunks.append({
                "text": tok,
                "metadata": {
                    "page": page_number,
                    "type": "section",
                    "has_image": False,
                }
            })
    return chunks

def parse_pdf_to_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []

    for page_idx, page in enumerate(doc):
        # Extract text blocks
        raw_blocks = page.get_text("blocks")
        text_blocks = []
        for b in raw_blocks:
            x0, y0, x1, y1, text, *_ = b
            if text.strip():
                text_blocks.append({
                    "bbox": (x0, y0, x1, y1),
                    "text": text.strip()
                })

        # Extract images
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))

            # OCR (with cleaning)
            raw_ocr = pytesseract.image_to_string(image)
            ocr_text = clean_ocr_text(raw_ocr)

            # Image bbox (PyMuPDF 1.23+)
            try:
                img_bbox = page.get_image_bbox(img)
            except Exception:
                continue

            # Caption + nearby text
            caption_block = find_nearest_caption(text_blocks, img_bbox)
            caption_text = caption_block["text"] if caption_block else ""
            nearby_text = collect_nearby_text(text_blocks, img_bbox)

            combined = "\n".join(t for t in [caption_text, ocr_text, nearby_text] if t)
            if combined:
                chunks.append({
                    "text": combined,
                    "metadata": {
                        "page": page_idx + 1,
                        "type": "figure",
                        "has_image": True,
                        "image_xref": xref,
                    }
                })

        # Section chunks
        page_text = "\n".join(tb["text"] for tb in text_blocks)
        chunks.extend(chunk_page_text(page_text, page_idx + 1))

    return chunks

# ---------------------------------------------------------------------------
# LLM: context-aware query expansion (FIX 1 & 2)
# ---------------------------------------------------------------------------

def generate_context_aware_queries(user_query, retrieved_docs, model="gpt-3.5-turbo"):
    context = "\n\n".join(retrieved_docs[:3])
    prompt = f"""
You are a ceramics and glaze chemistry assistant.

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
- are concise and single-topic.

List each question on a separate line without numbering.
"""
    messages = [
        {"role": "system", "content": "You generate grounded ceramics/glaze queries."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return [q.strip() for q in response.choices[0].message.content.split("\n") if q.strip()]

# ---------------------------------------------------------------------------
# LLM: final answer
# ---------------------------------------------------------------------------

def generate_final_answer(original_query, context, model="gpt-3.5-turbo"):
    prompt = f"""
You are a ceramics artist. Answer the user's question using ONLY the following excerpts from a ceramics glaze composition report. 
If the information is not in the excerpts, say so clearly.

Excerpts:
{context}

User question: {original_query}

Provide one concise, direct answer. Do not speculate beyond the excerpts.
"""
    messages = [
        {"role": "system", "content": "You answer questions based only on the provided document excerpts."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Build Chroma index with multimodal-aware chunks
# ---------------------------------------------------------------------------

PDF_PATH = "data/recipes.pdf"
COLLECTION_NAME = "glaze-collection"

chunks = parse_pdf_to_chunks(PDF_PATH)
docs = [c["text"] for c in chunks]
metadatas = [c["metadata"] for c in chunks]
ids = [str(i) for i in range(len(docs))]

embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME,
    embedding_function=embedding_function,
)

if collection.count() == 0:
    collection.add(ids=ids, documents=docs, metadatas=metadatas)

# ---------------------------------------------------------------------------
# Retrieval → expansion → joint retrieval → reranking → final answer
# (FIX 1, 3, 4)
# ---------------------------------------------------------------------------

ORIGINAL_QUERY = "What is the recipe for temmoku glaze?"
TOP_K_RERANK = 5
N_RESULTS_PER_QUERY = 10

# Initial retrieval driven by ORIGINAL_QUERY (FIX 1)
initial = collection.query(
    query_texts=[ORIGINAL_QUERY],
    n_results=5,
    include=["documents", "metadatas"],
)
retrieved_for_expansion = initial["documents"][0]

# Query expansion grounded in ORIGINAL_QUERY (FIX 1 & 2)
aug_queries = generate_context_aware_queries(ORIGINAL_QUERY, retrieved_for_expansion)

# Joint retrieval, but we’ll prefer section chunks via metadata (FIX 3)
joint_queries = [ORIGINAL_QUERY] + aug_queries

results = collection.query(
    query_texts=joint_queries,
    n_results=N_RESULTS_PER_QUERY,
    include=["documents", "metadatas"],
)

retrieved_docs_lists = results["documents"]
retrieved_meta_lists = results["metadatas"]

# Deduplicate with metadata
unique = {}
for docs_list, metas_list in zip(retrieved_docs_lists, retrieved_meta_lists):
    for d, m in zip(docs_list, metas_list):
        if d not in unique:
            unique[d] = m

unique_docs = list(unique.keys())
unique_metas = [unique[d] for d in unique_docs]

# Simple downweighting of figure chunks before reranking (FIX 4)
# We’ll pass all to cross-encoder but later bias toward sections.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[ORIGINAL_QUERY, d] for d in unique_docs]
scores = cross_encoder.predict(pairs)

# Apply a small penalty to figure chunks
for i, meta in enumerate(unique_metas):
    if meta.get("type") == "figure":
        scores[i] -= 0.2  # downweight figures slightly

# Prefer section chunks overall by sorting with adjusted scores (FIX 3 & 4)
top_idx = np.argsort(scores)[::-1][:TOP_K_RERANK]
top_docs = [unique_docs[i] for i in top_idx]

context = "\n\n".join(top_docs)
final_answer = generate_final_answer(ORIGINAL_QUERY, context)

print("Augmented queries:")
for q in aug_queries:
    print(f"  - {q}")

print("\nFinal answer:")
print(final_answer)