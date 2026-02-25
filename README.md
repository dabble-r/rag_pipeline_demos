# Advanced RAG Techniques

RAG (retrieval-augmented generation) over glaze and ceramics recipe PDFs: parse documents (text layer + optional OCR), index in ChromaDB, retrieve with optional query expansion, rerank with a cross-encoder, and answer with GPT.

---

## Key components

### `exp_query_final_cond.py` — RAG pipeline

Single-module pipeline used by the app and runnable from the CLI. It implements:

- **Parse → chunks**
  - **`parse_pdf_to_chunks_proposed`**: For each page, uses PyMuPDF text; if too short, falls back to Tesseract OCR. Splits into lines (preserving blanks), then applies a **dynamic parse strategy** (document-derived percentiles for alnum ratio, same-char run, repeat-pattern) to drop garbage lines while keeping valid content.
  - **Ingredient–quantity handling**: Lines in the form `"ingredient _______________ quantity"` are protected from being dropped and are parsed into `(ingredient, quantity)` pairs; these are stored in chunk metadata as `ingredient_quantities` (JSON string for ChromaDB).
- **Index**: Chunks are embedded with Sentence Transformers and stored in ChromaDB (collection name configurable).
- **Retrieval**: Initial query; then **conditional query expansion** — if the question is classified as OPEN or initial results are few, GPT generates extra queries; joint retrieval over original + expanded queries.
- **Rerank**: Cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks retrieved docs; optional distance threshold before rerank.
- **Answer**: Top reranked docs are passed to GPT (e.g. gpt-4o) for a single, cited answer.

**Main entrypoint:** `rag_pipeline(user_query, pdf_path, collection_name=..., top_k_rerank=..., ...)`  
**CLI:** `python exp_query_final_cond.py` (uses `OPENAI_API_KEY` from env and a default query/PDF; edit `__main__` to change).

---

### `app.py` — Streamlit UI

- **Glaze Recipe RAG** title and a text input for the user query.
- **Sidebar**: OpenAI API key (form), PDF upload (saved under `uploaded_pdfs/`).
- **Query button**: Runs `rag_pipeline(user_query, saved_pdf_path, openai_api_key=...)` and displays the returned answer. Requires key and an uploaded PDF.

**Run:** `streamlit run app.py`

---

## Setup

1. **Python 3.10+** and a virtualenv recommended.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Key deps: `chromadb`, `sentence-transformers`, `openai`, `PyMuPDF`, `pytesseract`, `streamlit`, `python-dotenv`.
3. **Optional — OCR:** For PDFs with little or no text layer, install Tesseract (e.g. `apt install tesseract-ocr` on Linux). OCR is used only when page text length is below a threshold.
4. **OpenAI:** Set `OPENAI_API_KEY` in `.env` or enter it in the app sidebar.

---

## Running

- **Web app:**  
  `streamlit run app.py`  
  Then open the URL (e.g. http://localhost:8501), add your API key and a PDF, type a query, and click Query.

- **CLI (script):**  
  Set `OPENAI_API_KEY` in `.env`, then:
  ```bash
  python exp_query_final_cond.py
  ```
  Edit the `rag_pipeline(...)` call in the `if __name__ == "__main__":` block to change the question or PDF path (e.g. `data/recipes_kc.pdf`).

---

## Project layout

| Path | Purpose |
|------|--------|
| `exp_query_final_cond.py` | RAG pipeline (parse, index, expand, retrieve, rerank, answer). |
| `app.py` | Streamlit UI that calls `rag_pipeline`. |
| `requirements.txt` | Python dependencies (ChromaDB, sentence-transformers, OpenAI, PyMuPDF, Tesseract, Streamlit, etc.). |
| `data/` | Sample PDFs (e.g. glaze recipes); gitignored. |
| `.env` | `OPENAI_API_KEY`; gitignored. |
| `md_chat/` | Notes (e.g. `tesseraact_ocr.md` on OCR/parse strategy and ingredient-quantity handling; `parse_text_pdf.md`). |

---

## Configuration

Pipeline behavior is controlled by `rag_pipeline` arguments, for example:

- `collection_name` — ChromaDB collection.
- `top_k_rerank` — Number of docs kept after rerank (default 5).
- `initial_n_results`, `n_results_per_query` — Retrieval sizes.
- `expand_threshold` — Expand queries if initial doc count is below this (and/or if question is OPEN).
- `min_sim_threshold` — Optional ChromaDB distance filter before rerank.
- `cross_encoder_model` — Reranker model name.
- `openai_api_key` — Override for API key.

Parse/OCR options live inside the pipeline (e.g. `use_ocr_fallback`, `dpi`, `ocr_min_len`, and the dynamic-strategy percentiles in `parse_pdf_to_chunks_proposed` / `_apply_dynamic_strategy`).
