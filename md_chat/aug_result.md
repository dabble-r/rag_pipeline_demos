# Augmented Queries, Retrieved Docs, and Final-Answer Proposal

## Summary of `exp_query_context_aware.py`

### 1. Augmented queries (aug_queries)

- **Source**: `generate_context_aware_queries(user_query, retrieved_docs)` (lines 14–55).
- **Inputs**:
  - **user_query**: The initial search query, e.g. `"What sectors of the tech industry does Microsoft operate in?"`.
  - **retrieved_docs**: The top 5 documents from a first retrieval with that query; the function uses only the first 3 to build context.
- **Behavior**: The LLM is prompted to produce **up to five** follow-up questions that:
  - Explore different angles of the same topic,
  - Stay grounded in the retrieved content,
  - Avoid speculation,
  - Are concise and single-topic,
  - Help retrieve more relevant information.
- **Output**: A list of strings, e.g. `["question1", "question2", ...]` (no numbering, one per line).

### 2. Original query used for the “final answer”

- **Variable**: `original_query` (lines 157–159).
- **Example**: `"Which sector of the tech industry shows the highest potential for growth for Microsoft over the next 5 years?"`
- This is the **user-facing question** we want the system to answer in one final response.

### 3. Joint queries and retrieval

- **joint_query** (line 172): `[original_query] + aug_queries` — the original user question plus all augmented questions.
- **Retrieval** (lines 174–177):  
  `chroma_collection.query(query_texts=joint_query, n_results=5, include=["documents", "embeddings"])`
- **results["documents"]**: A list of lists:
  - One list per query in `joint_query`.
  - Each inner list has up to 5 document chunks (strings).
- **Deduplication** (lines 180–184): All chunks from all queries are merged and deduplicated into a single set `unique_documents` (so the same chunk from multiple queries is only kept once).

### 4. What the script does *not* do

- It does **not** call the LLM to synthesize the retrieved chunks into **one final answer** to `original_query`. It only prints the augmented queries and the retrieved documents per query.

---

## Proposal: Query LLM for One Final Response

### Goal

Given:

- `original_query`: the user’s question (e.g. growth potential by sector),
- `aug_queries`: context-aware expansion questions,
- `unique_documents` (or the equivalent list): deduplicated chunks from retrieval over `joint_query`,

we want **one** LLM call that uses this context to produce a single, coherent answer to `original_query`.

### Design choices

1. **Context size**: The full `unique_documents` set can be large. Options:
   - **Option A**: Use all deduplicated docs (may hit context limits for long reports).
   - **Option B (recommended)**: Rank or re-rank chunks by relevance to `original_query` (e.g. embedding similarity or a reranker), then pass the top‑k (e.g. top 10–15) to the LLM.
   - **Option C**: Concatenate up to a token budget (e.g. 4k tokens) and truncate.

2. **What to pass to the LLM**:
   - **Must have**: `original_query` (the question to answer) and a single combined context string built from the chosen chunks.
   - **Optional**: A short note that the context comes from an annual report and that the model should cite or stay grounded in it; and instruction to say “not found” if the context does not support an answer.

3. **Output**: One final answer string (and optionally a confidence or “no information” flag).

### Proposed implementation (pseudocode)

```python
def generate_final_answer(original_query: str, unique_documents: set, model: str = "gpt-3.5-turbo") -> str:
    # 1. Optionally limit/rank documents (e.g. top 12 by relevance to original_query)
    doc_list = list(unique_documents)
    # If you have a reranker or want to re-query Chroma with only original_query and take top_k:
    # doc_list = get_top_k_for_query(original_query, doc_list, k=12)
    context = "\n\n---\n\n".join(doc_list[:12])  # cap to avoid token limit

    prompt = f"""You are a financial analysis assistant. Answer the user's question using ONLY the following excerpts from a company annual report. If the information is not in the excerpts, say so clearly.

Excerpts:
{context}

User question: {original_query}

Provide one concise, direct answer. Do not speculate beyond the excerpts."""

    messages = [
        {"role": "system", "content": "You answer questions based only on the provided document excerpts."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()
```

### Where to call it in `exp_query_context_aware.py`

- After building `unique_documents` (after line 184).
- Convert `unique_documents` to a list (or use a ranked subset), then call:
  - `final_answer = generate_final_answer(original_query, unique_documents)`  
- Print or return `final_answer` as the single final response to the user.

### Optional improvement: relevance-based trimming

- Re-embed `original_query` and each chunk in `unique_documents`, compute similarity (e.g. cosine), sort by score, and pass only the top 10–12 chunks to `generate_final_answer`. That keeps the prompt smaller and focuses the LLM on the most relevant passages.

---

## Analysis of `reranking.py`

### How reranking works

1. **Retrieval** (lines 65–69, 106–108): Single query or multiple queries (`[original_query] + generated_queries`) are sent to Chroma; `n_results=10` per query; results include `documents` and `embeddings`.

2. **Deduplication** (lines 111–117): All retrieved chunks are merged and deduplicated into a list `unique_documents`.

3. **Cross-encoder reranking** (lines 76–77, 120–124):
   - **Model**: `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` from `sentence_transformers`. It scores (query, document) pairs for relevance.
   - **Pairs**: For each doc in `unique_documents`, build `[original_query, doc]`.
   - **Scores**: `scores = cross_encoder.predict(pairs)` — one relevance score per document.

4. **Top-k selection** (lines 133–134):  
   `top_indices = np.argsort(scores)[::-1][:5]`  
   `top_documents = [unique_documents[i] for i in top_indices]`  
   So the **top 5** docs by cross-encoder score are kept.

5. **Context and final answer** (lines 136–137, 140–165):  
   `context = "\n\n".join(top_documents)` is passed to `generate_multi_query(original_query, context)`, which calls the OpenAI chat API to produce the final answer from that context only.

### Takeaways for integration

- Reranking is done **after** retrieval and deduplication, **before** building the context for the LLM.
- The **query used for reranking** is always `original_query` (the user’s question), so the final context is optimized for that question rather than for the expanded queries.
- Cross-encoder gives a better relevance signal than embedding similarity alone; using it keeps the prompt small (e.g. top 5) while staying focused on the most relevant passages.

---

## Integration: Rerank Retrieved Docs and Aug Queries → One Final Response

### End-to-end flow (with reranking)

1. **exp_query_context_aware.py** (existing):  
   Initial retrieval with `query` → get `aug_queries` from `generate_context_aware_queries` → build `joint_query = [original_query] + aug_queries` → Chroma retrieval for each → deduplicate to `unique_documents`.

2. **New: Rerank** (from reranking.py):  
   Convert `unique_documents` to a list. For each doc, score the pair `(original_query, doc)` with the cross-encoder. Sort by score descending and take the top **k** (e.g. 5, as in reranking.py, or 8–10 for more context). Call this list `top_documents`.

3. **New: Final answer** (from aug_result proposal):  
   Build `context = "\n\n".join(top_documents)` and call `generate_final_answer(original_query, context)` (or pass `top_documents` and join inside the function). The LLM returns **one** final response using only the reranked chunks.

### Proposed code additions for `exp_query_context_aware.py`

**Step 1 — Rerank (after deduplication, ~line 184):**

```python
import numpy as np
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
unique_doc_list = list(unique_documents)
pairs = [[original_query, doc] for doc in unique_doc_list]
scores = cross_encoder.predict(pairs)
top_k = 5  # or 8–10
top_indices = np.argsort(scores)[::-1][:top_k]
top_documents = [unique_doc_list[i] for i in top_indices]
```

**Step 2 — Build context and generate one final response:**

```python
context = "\n\n".join(top_documents)
final_answer = generate_final_answer(original_query, context, model="gpt-3.5-turbo")
print("Final answer:", final_answer)
```

`generate_final_answer` here takes `(original_query, context)` where `context` is the pre-joined string of `top_documents` (no set/list of docs needed; the reranker already chose the best chunks).

### Dependency

- **sentence_transformers** (already in `requirements.txt`): provides `CrossEncoder` and the `cross-encoder/ms-marco-MiniLM-L-6-v2` model (downloaded on first use).

### Summary of the integrated pipeline

| Step | Where | Action |
|------|--------|--------|
| 1 | exp_query_context_aware | Initial retrieval → context-aware aug_queries → joint_query retrieval → deduplicate → `unique_documents` |
| 2 | **New (reranking)** | Cross-encoder score each `(original_query, doc)` for `doc` in `unique_documents` → take top_k → `top_documents` |
| 3 | **New (final answer)** | `context = "\n\n".join(top_documents)` → LLM with `original_query` + `context` → **one final response** |

This gives: **augmented queries** for broad retrieval, **reranking** so only the most relevant chunks to the user question are kept, and **one** LLM call for the final answer.

---

## Summary

| Item | Description |
|------|-------------|
| **aug_queries** | LLM-generated follow-up questions grounded in the first retrieval; used to expand retrieval. |
| **retrieved_documents** | Per-query lists of 5 chunks from Chroma for `joint_query`; then merged and deduplicated. |
| **unique_documents** | Set of all unique chunks that can be used as context for the final answer. |
| **reranking.py** | Uses a cross-encoder (ms-marco-MiniLM-L-6-v2) to score (original_query, doc) pairs, takes top 5, then one LLM call for the final answer. |
| **Integrated solution** | In `exp_query_context_aware.py`: after deduplication → rerank with cross-encoder by `original_query` → take top_k docs → build context → `generate_final_answer(original_query, context)` for **one final response**. |

Implementing the reranking step and `generate_final_answer` in `exp_query_context_aware.py` (as in the “Proposed code additions” above) gives: augmented queries for retrieval, reranked docs for relevance, and one LLM-generated final answer.
