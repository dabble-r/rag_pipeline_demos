"""
FULL RAG PIPELINE WITH:
- Layout-aware PDF parsing
- Revised OCR cleaning
- Image detection but NO image embedding
- Text-only chunks (headings, captions, paragraphs, OCR text)
- Metadata-aware retrieval
- Query expansion using ORIGINAL_QUERY
- Cross-encoder reranking with figure downweighting
- MINIMUM SIMILARITY THRESHOLD BEFORE RERANKING  (Solution 4)
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
# OCR CLEANING (REVISED)
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> str:
    mineral_terms = [
        "feldspar", "whiting", "silica", "kaolin", "ball clay", "iron",
        "copper", "cobalt", "manganese", "rutile", "zircon", "tin"
    ]

    lines = text.splitlines()
    cleaned = []

    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if len(raw) < 3:
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
# TEXT CHUNKING HELPERS
# ---------------------------------------------------------------------------

def chunk_page_text(page_text, page_number):
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
                    "has_image": False
                }
            })
    return chunks


def find_nearest_caption(text_blocks, img_bbox):
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
    ix0, iy0, ix1, iy1 = img_bbox
    collected = []
    for tb in text_blocks:
        tx0, ty0, tx1, ty1 = tb["bbox"]
        if abs(ty0 - iy1) < threshold or abs(iy0 - ty1) < threshold:
            collected.append(tb["text"])
    return "\n".join(collected)


# ---------------------------------------------------------------------------
# MAIN INGESTION FUNCTION
# ---------------------------------------------------------------------------

def parse_pdf_to_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []

    for page_idx, page in enumerate(doc):
        raw_blocks = page.get_text("blocks")
        text_blocks = []
        for b in raw_blocks:
            x0, y0, x1, y1, text, *_ = b
            if text.strip():
                text_blocks.append({
                    "bbox": (x0, y0, x1, y1),
                    "text": text.strip()
                })

        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))

            raw_ocr = pytesseract.image_to_string(image)
            ocr_text = clean_ocr_text(raw_ocr)

            try:
                img_bbox = page.get_image_bbox(img)
            except Exception:
                continue

            caption_block = find_nearest_caption(text_blocks, img_bbox)
            caption_text = caption_block["text"] if caption_block else ""
            nearby_text = collect_nearby_text(text_blocks, img_bbox)

            combined = "\n".join(t for t in [caption_text, ocr_text, nearby_text] if t)

            if combined.strip():
                chunks.append({
                    "text": combined,
                    "metadata": {
                        "page": page_idx + 1,
                        "type": "figure_text",
                        "has_image": False
                    }
                })

        page_text = "\n".join(tb["text"] for tb in text_blocks)
        chunks.extend(chunk_page_text(page_text, page_idx + 1))

    return chunks


# ---------------------------------------------------------------------------
# LLM: context-aware query expansion
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

Generate up to five grounded follow-up questions.
Each on its own line.
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
You are a ceramics artist. Answer the user's question using ONLY the following excerpts.
If the information is not in the excerpts, say so clearly.

Excerpts:
{context}

User question: {original_query}

Provide one concise answer.
"""
    messages = [
        {"role": "system", "content": "You answer using only provided excerpts."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Build Chroma index with text-only chunks
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
# Retrieval → expansion → reranking → final answer
# ---------------------------------------------------------------------------

ORIGINAL_QUERY = "What is the recipe for temmoku glaze?"
TOP_K_RERANK = 5
N_RESULTS_PER_QUERY = 10
MIN_SIM_THRESHOLD = 0.35   # <<< SOLUTION 4

# Initial retrieval
initial = collection.query(
    query_texts=[ORIGINAL_QUERY],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)
retrieved_for_expansion = initial["documents"][0]

# Query expansion
aug_queries = generate_context_aware_queries(ORIGINAL_QUERY, retrieved_for_expansion)

# Joint retrieval
joint_queries = [ORIGINAL_QUERY] + aug_queries
results = collection.query(
    query_texts=joint_queries,
    n_results=N_RESULTS_PER_QUERY,
    include=["documents", "metadatas", "distances"],
)

# Deduplicate
unique = {}
for docs_list, metas_list, dist_list in zip(
    results["documents"], results["metadatas"], results["distances"]
):
    for d, m, dist in zip(docs_list, metas_list, dist_list):
        if d not in unique:
            unique[d] = {"meta": m, "dist": dist}

# Apply MINIMUM SIMILARITY THRESHOLD
filtered_docs = []
filtered_metas = []

for d, info in unique.items():
    if info["dist"] < MIN_SIM_THRESHOLD:
        filtered_docs.append(d)
        filtered_metas.append(info["meta"])

# If filtering removes everything, fall back to all docs
if not filtered_docs:
    filtered_docs = list(unique.keys())
    filtered_metas = [unique[d]["meta"] for d in filtered_docs]

# Rerank
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[ORIGINAL_QUERY, d] for d in filtered_docs]
scores = cross_encoder.predict(pairs)

# Downweight figure_text
for i, meta in enumerate(filtered_metas):
    if meta.get("type") == "figure_text":
        scores[i] -= 0.2

top_idx = np.argsort(scores)[::-1][:TOP_K_RERANK]
top_docs = [filtered_docs[i] for i in top_idx]

context = "\n\n".join(top_docs)
final_answer = generate_final_answer(ORIGINAL_QUERY, context)

print("Augmented queries:")
for q in aug_queries:
    print(" -", q)

print("\nFinal answer:")
print(final_answer)