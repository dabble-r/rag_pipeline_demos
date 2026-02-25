
def rag_pipeline(
    user_query: str,
    *,
    pdf_path: str = "",
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
    client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))

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

    raw_groups = parse_pdf_to_chunks(pdf_path, user_query)
    # print("raw groups:", raw_groups)


    docs = []
    metas = []
    for g in raw_groups:
        page = g["metadata"]["page"]
        # print("page: ", page)
        for ch in chunk_text(g["text"], page):
            docs.append(ch["text"])
            metas.append(ch["metadata"])

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
    result = rag_pipeline(
        "answer",
        "To experiment with glaze recipes, which recipe is the best starting point?",
    )
    print("result: ", result)