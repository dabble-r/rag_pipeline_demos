
def rag_pipeline(
    user_query: str,
    pdf_path: str,
    collection_name: str = "glaze-collection",
    top_k_rerank: int = 5,
    n_results_per_query: int = 10,
    initial_n_results: int = 5,
    expand_threshold: int = 2,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    min_sim_threshold: float | None = None,
    openai_api_key: str | None = None,
):
    """
    RAG pipeline: parse PDF (text + optional OCR) → index → retrieve → optional expansion → rerank → answer.

    Args:
        task: Reserved for future use (e.g. model routing).
        user_query: The user's question.
        pdf_path: Path to the PDF file.
        collection_name: ChromaDB collection name.
        top_k_rerank: Number of top docs to keep after cross-encoder rerank.
        n_results_per_query: Max results per query when running joint queries.
        initial_n_results: Results from initial query (used for expansion decision and context).
        expand_threshold: Min number of initial docs below which we expand (if not OPEN).
        cross_encoder_model: Sentence-transformers model for reranking.
        min_sim_threshold: If set, filter retrieved docs by distance below this before rerank (ChromaDB distance).
        openai_api_key: Override for OpenAI API key (default: from env OPENAI_API_KEY).
    """

    
    import re
    import os
    import io
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
    os.environ["OPENAI_API_KEY"] = openai_api_key
    client = OpenAI(api_key=openai_api_key)

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

    def classify_query_type(user_query: str) -> str:
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


    def should_expand_query(user_query, initial_docs, threshold=None):
        threshold = expand_threshold if threshold is None else threshold
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
    # DEPRECATED: USE parse_pdf_to_chunks_proposed INSTEAD
    # ---------------------------------------------------------------------------

    def parse_pdf_to_chunks(
      pdf_path: str,
      user_query=None,
      use_ocr_fallback: bool = True,
      dpi: int = 150,
      ocr_min_len: int = 10,
      ) -> list[dict]:
          """
          Parse a PDF into chunks (text layer + optional OCR for empty/short pages).
          Merges behavior of pdf_to_single_string and parse_pdf_to_chunks: correct
          page API, OCR when needed, and chunk list with metadata for RAG.

          Args:
              pdf_path: Path to the PDF file.
              user_query: Unused; kept for API compatibility with callers that pass it.
              use_ocr_fallback: If True, run OCR on a page when its text layer length < ocr_min_len.
              dpi: Resolution for rendering pages when running OCR.
              ocr_min_len: Run OCR when page text length is below this (default 20).

          Returns:
              List of {"text": str, "metadata": {"page": int, "type": str}}.
              For a single string: "\\n\\n".join(c["text"] for c in chunks).
          """
          doc = fitz.open(pdf_path)
          chunks = []

          for page_idx in range(len(doc)):
              page = doc[page_idx]
              raw = page.get_text()
              # print("raw page sample: ", raw)

              text = (raw or "").strip()
              #print("text page sample: ", text)

              if use_ocr_fallback and len(text) < ocr_min_len:
                  try:
                      pix = page.get_pixmap(dpi=dpi)
                      img_bytes = pix.tobytes("png")
                      image = Image.open(io.BytesIO(img_bytes))
                      text = (pytesseract.image_to_string(image) or "").strip()
                      # print("ocr text page sample: ", text)
                  except Exception:
                      pass

              if text:
                  chunks.append({
                      "text": text,
                      "metadata": {"page": page_idx + 1, "type": "section"},
                  })

          doc.close()
          return chunks


    # ---------------------------------------------------------------------------
    # DYNAMIC PARSE STRATEGY (document-derived thresholds, no hardcoding)
    # ---------------------------------------------------------------------------

    def _clean_line_for_parse(line: str, collapse_internal_punct: bool = True) -> str:
        """Generic line cleaning: strip, collapse leading/trailing punctuation, normalize spaces."""
        import re
        s = line.strip()
        if not s:
            return s
        # Collapse internal whitespace to single space
        s = re.sub(r"\s+", " ", s)
        # Strip leading/trailing non-letter sequences (punctuation/symbols)
        s = re.sub(r"^[\W_]+", "", s)
        s = re.sub(r"[\W_]+$", "", s)
        if collapse_internal_punct:
            # Collapse runs of same non-alphanumeric character only
            s = re.sub(r"([\W_])\1+", r"\1", s)
        return s.strip()

    def _line_features(line: str) -> dict:
        """Per-line features for document-level outlier detection."""
        if not line:
            return {"length": 0, "alnum_ratio": 0.0, "max_same_run": 0, "repeat_score": 0.0}
        length = len(line)
        alnum = sum(1 for c in line if c.isalnum())
        alnum_ratio = alnum / length if length else 0.0
        # Max run of same character
        max_run = 0
        run = 0
        prev = None
        for c in line:
            if c == prev:
                run += 1
            else:
                run = 1
                prev = c
            max_run = max(max_run, run)
        # Repeat-pattern score: max count of any 2-char substring / length (bounded)
        repeat_score = 0.0
        if length >= 2:
            from collections import Counter
            pairs = [line[i : i + 2] for i in range(length - 1)]
            if pairs:
                cnt = Counter(pairs)
                most = max(cnt.values()) if cnt else 0
                repeat_score = most / length
        return {"length": length, "alnum_ratio": alnum_ratio, "max_same_run": max_run, "repeat_score": repeat_score}

    def _is_ingredient_quantity_line(cleaned: str) -> bool:
        """True if line looks like 'ingredient filler quantity' (at least one letter, ends with digits optional decimal/%)."""
        import re
        if not cleaned or not cleaned.strip():
            return False
        return bool(re.search(r"[A-Za-z]", cleaned) and re.search(r"\d+\.?\d*%?\s*$", cleaned))

    def _parse_ingredient_quantity_line(cleaned: str):
        """If line matches 'ingredient filler quantity', return (ingredient, quantity); else None."""
        import re
        m = re.search(r"^(.+?)[\W_]+(\d+\.?\d*%?)\s*$", cleaned)
        if not m:
            return None
        left = m.group(1).strip()
        right = m.group(2).strip()
        if not left or not re.search(r"[A-Za-z]", left):
            return None
        return (left, right)

    def _apply_dynamic_strategy(
        lines_by_page: list,
        min_len: int = 1,
        p_ratio_low: float = 5.0,
        p_same_run_high: float = 95.0,
        use_repeat_heuristic: bool = True,
        p_repeat_high: float = 95.0,
        collapse_internal_punct: bool = True,
    ) -> list[dict]:
        """
        Apply document-derived thresholds: collect features from all lines, compute percentiles,
        drop outlier lines, reassemble per page. Returns list of {"text": str, "metadata": {"page": int, "type": str}}.
        Collapse (internal punctuation) is applied before computing features so "ingredient _____ quantity"
        lines are not dropped as garbage.
        """
        # Collect (page_idx, line, is_blank) for all lines
        all_items = []
        for page_idx, line_list in lines_by_page:
            for line, is_blank in line_list:
                all_items.append((page_idx, line, is_blank))

        # Clean non-blank lines and compute features
        non_blank_features = []
        cleaned_per_item = []
        for page_idx, line, is_blank in all_items:
            if is_blank:
                cleaned_per_item.append((page_idx, "", True))
                continue
            cleaned = _clean_line_for_parse(line, collapse_internal_punct=collapse_internal_punct)
            cleaned_per_item.append((page_idx, cleaned, False))
            if cleaned:
                feats = _line_features(cleaned)
                non_blank_features.append(feats)

        if not non_blank_features:
            # No non-blank lines; return one chunk per page with empty or original text
            pages_seen = {}
            for page_idx, _c, _b in cleaned_per_item:
                pages_seen.setdefault(page_idx, [])
            return [{"text": "", "metadata": {"page": p + 1, "type": "section"}} for p in sorted(pages_seen.keys())]

        # Document-level percentiles
        ratios = [f["alnum_ratio"] for f in non_blank_features]
        same_runs = [f["max_same_run"] for f in non_blank_features]
        repeat_scores = [f["repeat_score"] for f in non_blank_features]
        p5_ratio = float(np.percentile(ratios, p_ratio_low))
        p95_same = float(np.percentile(same_runs, p_same_run_high))
        p95_repeat = float(np.percentile(repeat_scores, p_repeat_high)) if use_repeat_heuristic else 1.0

        # Mark each item keep/drop (recompute features for cleaned lines to align with percentiles)
        kept_items = []
        for page_idx, cleaned, is_blank in cleaned_per_item:
            if is_blank:
                kept_items.append((page_idx, "", True, True))
                continue
            if not cleaned:
                kept_items.append((page_idx, cleaned, False, False))
                continue
            feats = _line_features(cleaned)
            drop = (
                feats["alnum_ratio"] < p5_ratio
                or feats["max_same_run"] > p95_same
                or feats["length"] < min_len
                or (use_repeat_heuristic and feats["repeat_score"] > p95_repeat)
            )
            # Protect "ingredient _____ quantity" lines: do not drop even if heuristics say garbage
            if _is_ingredient_quantity_line(cleaned):
                drop = False
            kept_items.append((page_idx, cleaned, False, not drop))

        # Reassemble per page: group by page_idx, join kept lines preserving blanks
        from itertools import groupby
        page_blocks = []
        for page_idx, group in groupby(kept_items, key=lambda x: x[0]):
            page_lines = []
            ingredient_quantities = []
            for _, cleaned, is_blank, keep in group:
                if is_blank:
                    page_lines.append("\n\n")
                elif keep:
                    page_lines.append(cleaned)
                    pair = _parse_ingredient_quantity_line(cleaned)
                    if pair:
                        ingredient_quantities.append(pair)
            text = "\n".join(page_lines).replace("\n\n\n", "\n\n").strip()
            meta = {"page": page_idx + 1, "type": "section"}
            if ingredient_quantities:
                import json
                meta["ingredient_quantities"] = json.dumps(ingredient_quantities)
            if text:
                page_blocks.append({"text": text, "metadata": meta})

        return page_blocks

    def parse_pdf_to_chunks_proposed(
        pdf_path: str,
        user_query=None,
        use_ocr_fallback: bool = True,
        dpi: int = 150,
        ocr_min_len: int = 20,
        min_line_len: int = 1,
        p_ratio_low: float = 5.0,
        p_same_run_high: float = 95.0,
        use_repeat_heuristic: bool = True,
        p_repeat_high: float = 95.0,
        collapse_internal_punct: bool = True,
    ) -> list[dict]:
        """
        Parse PDF into chunks using merged flow (text layer + OCR fallback) then apply
        dynamic parse strategy: document-derived thresholds, generic cleaning, no hardcoding.
        Same return shape as parse_pdf_to_chunks; use this for robust OCR/post-parse.
        """
        doc = fitz.open(pdf_path)
        # 1) Get page texts (same as parse_pdf_to_chunks_merged)
        page_texts = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            raw = page.get_text()
            text = (raw or "").strip()
            if use_ocr_fallback and len(text) < ocr_min_len:
                try:
                    pix = page.get_pixmap(dpi=dpi)
                    img_bytes = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_bytes))
                    text = (pytesseract.image_to_string(image) or "").strip()
                    # print("ocr text page sample: ", text)
                except Exception:
                    pass
            page_texts.append((page_idx, text))
        doc.close()

        # 2) Split each page into lines and record blanks (structure)
        lines_by_page = []
        for page_idx, text in page_texts:
            line_list = []
            for raw_line in text.splitlines():
                # print("raw line sample: ", raw_line)
                is_blank = not raw_line.strip()
                line_list.append((raw_line, is_blank))
            lines_by_page.append((page_idx, line_list))

        # 3) Apply dynamic strategy and return chunks
        return _apply_dynamic_strategy(
            lines_by_page,
            min_len=min_line_len,
            p_ratio_low=p_ratio_low,
            p_same_run_high=p_same_run_high,
            use_repeat_heuristic=use_repeat_heuristic,
            p_repeat_high=p_repeat_high,
            collapse_internal_punct=collapse_internal_punct,
        )

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

    raw_groups = parse_pdf_to_chunks_proposed(pdf_path, user_query)
    # print("raw groups:", raw_groups)

    docs = []
    metas = []
    for g in raw_groups:
        page = g["metadata"]["page"]
        for ch in chunk_text(g["text"], page):
            docs.append(ch["text"])
            metas.append({**ch["metadata"], **g["metadata"]})

    # for i, d in enumerate(docs[:3]):
    #    print("CHUNK", i, "\n", d[:500], "\n---\n")

    ids = [str(i) for i in range(len(docs))]

    if not docs:
        raise SystemExit(
            "No documents to add: parse_pdf_to_chunks produced no chunks (empty PDF, no matching sections, or no text in spans)."
        )

    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        collection_name,
        embedding_function=embedding_function,
    )

    if collection.count() == 0:
        collection.add(ids=ids, documents=docs, metadatas=metas)


    # ---------------------------------------------------------------------------
    # RETRIEVAL → CONDITIONAL EXPANSION
    # ---------------------------------------------------------------------------

    initial = collection.query(
        query_texts=[user_query],
        n_results=initial_n_results,
        include=["documents", "distances"] if min_sim_threshold is not None else ["documents"],
    )
    initial_docs = initial["documents"][0]

    expand = should_expand_query(user_query, initial_docs)

    if expand:
        raw_expansions = generate_context_aware_queries(user_query, initial_docs)
        aug_queries = raw_expansions
        joint_queries = [user_query] + aug_queries
    else:
        aug_queries = []
        joint_queries = [user_query]

    results = collection.query(
        query_texts=joint_queries,
        n_results=n_results_per_query,
        include=["documents", "distances"] if min_sim_threshold is not None else ["documents"],
    )

    unique_docs = []
    seen = set()
    distances_by_doc = {}
    if min_sim_threshold is not None and "distances" in results:
        for doc_list, dist_list in zip(results["documents"], results["distances"]):
            for d, dist in zip(doc_list, dist_list):
                if d not in seen:
                    seen.add(d)
                    distances_by_doc[d] = dist
                    unique_docs.append(d)
        unique_doc_list = [d for d in unique_docs if distances_by_doc[d] < min_sim_threshold]
        if not unique_doc_list:
            unique_doc_list = unique_docs
    else:
        for doc_list in results["documents"]:
            for d in doc_list:
                if d not in seen:
                    unique_docs.append(d)
                    seen.add(d)
        unique_doc_list = list(unique_docs)


    # ---------------------------------------------------------------------------
    # CROSS-ENCODER RERANKING
    # ---------------------------------------------------------------------------

    cross_encoder = CrossEncoder(cross_encoder_model)
    pairs = [[user_query, doc] for doc in unique_doc_list]
    scores = cross_encoder.predict(pairs)

    top_indices = np.argsort(scores)[::-1][:top_k_rerank]
    top_documents = [unique_doc_list[i] for i in top_indices]


    # ---------------------------------------------------------------------------
    # FINAL ANSWER
    # ---------------------------------------------------------------------------

    context = "\n\n".join(top_documents)
    final_answer = generate_final_answer(user_query, context)

    print("\nAugmented queries:")
    for q in aug_queries:
        print(" -", q)

    print("\nFinal answer:")
    return final_answer


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    result = rag_pipeline(
        "Which materials tend to produce crystals in a glaze?",
        "data/ceramics.pdf",
        openai_api_key=openai_api_key,
    )
    print("result: ", result)