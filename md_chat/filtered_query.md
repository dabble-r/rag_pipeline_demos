# Analysis: Why `parse_pdf_to_chunks` Produces No Chunks

This document analyzes `exp_query_mulit_filtered.py` and identifies flaws that can lead to **zero chunks** (and thus "No documents to add"). No code changes are made here; solutions are proposed for maintainers to implement.

---

## 1. Flow Summary

- **Source:** `parse_pdf_to_chunks(PDF_PATH, ORIGINAL_QUERY)` → list of `{ "text": str, "metadata": {...} }`.
- **Downstream:** Each group’s `text` is split by `chunk_text()` → `docs` and `metas`. If `raw_groups` is empty, `docs` stays empty and ChromaDB add is skipped (or the script exits with the current guard).
- **Conclusion:** The only way to get "No documents to add" from this pipeline is **`parse_pdf_to_chunks` returning an empty list**. The following flaws explain why that can happen.

---

## 2. Flaws in `parse_pdf_to_chunks`

### 2.1 No text in spans (PDF structure / image-only)

**What happens:** `page.get_text("rawdict")` returns a hierarchy `blocks → lines → spans`. Spans are only kept if `span.get("text")` is non-empty after strip. If the PDF is image-only (scanned), has no text layer, or uses an odd structure, `raw["blocks"]` may be empty or all spans may have no `"text"` key. Then `spans` stays empty for every page and no chunk is ever created.

**Why it’s a flaw:** There is no fallback (e.g. OCR) when rawdict yields no text. The code assumes a text-heavy PDF with a normal block/line/span layout.

---

### 2.2 Heading pattern never matches (query vs. document wording)

**What happens:** The "section" pattern is built from the user query:

- Words are extracted with `re.findall(r"[a-zA-Z]+", q)` and filtered to `len(w) > 2`.
- If there is at least one such word, `heading_pattern = re.compile(r"(" + "|".join(words) + r")", re.I)` (e.g. for "What is the recipe for temmoku glaze?" → `(what|recipe|for|temmoku|glaze)`).
- If there are no words longer than 2 characters, fallback is `re.compile(r"glaze$", re.I)` (match only lines that **end** with "glaze").

A section is only started when **some span's text** matches this pattern. If the PDF uses different wording (e.g. "Temmoku", "Recipes", "Formulation" but never "temmoku" or "glaze", or headings like "Glaze Recipe" that do not **end** with "glaze"), no span ever matches. Then `inside_section` is never set to `True`, no line is ever added to `current_group`, and chunks stay empty.

**Why it’s a flaw:** The logic assumes section headings contain one of the query words (or, in fallback, that headings literally end with "glaze"). It is brittle for varied PDF wording and for short queries that yield the strict `glaze$` fallback.

---

### 2.3 Single-heading sections discarded when next line doesn’t match cone/recipe

**What happens:** After a heading matches, the code only keeps adding lines that match:

- `cone_pattern`: `r"cone\s*\d"`, or  
- `recipe_line_pattern`: `r"[A-Za-z].*\d"` (at least one letter and at least one digit).

Any other line (e.g. "Ingredients:", "Instructions:", "Notes:") is treated as "unrelated":

```python
if inside_section:
    if cone_pattern.search(t) or recipe_line_pattern.search(t):
        current_group.append(t)
        continue
    # Stop when unrelated text appears
    if len(current_group) > 1:
        chunks.append(...)
    current_group = []
    inside_section = False
```

If the **only** line in the section so far is the heading (`len(current_group) == 1`), the block does **not** append to `chunks`; it only clears `current_group` and sets `inside_section = False`. So:

- The heading-only section is discarded.
- Subsequent recipe lines (e.g. "Custer Feldspar 45") are ignored because we're no longer `inside_section` and they don't match the heading pattern.

**Why it’s a flaw:** One intervening label or non-recipe line (e.g. "Ingredients:") after the heading causes the whole section to be dropped and later recipe lines to be skipped. The design assumes headings are immediately followed by cone/recipe lines with no such labels in between.

---

### 2.4 Fallback pattern `glaze$` is too strict

**What happens:** When the query has no word longer than 2 characters, the fallback is `heading_pattern = re.compile(r"glaze$", re.I)`. So only lines that **end** with the word "glaze" start a section. Lines like "Glaze Recipe", "Glaze safety", or "Temmoku glaze base" (if the span is different) may not match if the span text doesn't end with "glaze".

**Why it’s a flaw:** The fallback is unnecessarily strict; many valid headings contain "glaze" but don't end with it, so they never start a section.

---

### 2.5 No diagnostic output

**What happens:** When zero chunks are returned, the script exits with a single message. There is no logging of: page count, number of blocks/spans per page, whether any span matched the heading pattern, or whether sections were dropped due to "unrelated" lines.

**Why it’s a flaw:** Hard to tell whether the failure is due to no text, no heading match, or section-dropping logic without adding instrumentation.

---

## 3. Proposed Solutions (no code changes in this repo)

### 3.1 When rawdict yields no text

- **Option A:** Add an OCR path when `rawdict` gives no (or very little) text per page (e.g. extract page as image, run Tesseract or similar, then run the same grouping logic on OCR lines).
- **Option B:** At least detect "no spans with text" and raise or log a clear message (e.g. "No text in PDF (consider OCR)").

---

### 3.2 Make heading detection more robust

- **Option A:** Broaden fallback: e.g. use `r"\bglaze\b"` (word boundary) instead of `glaze$`, or a small set of known section keywords (e.g. "glaze", "recipe", "cone") so at least one match is likely.
- **Option B:** Allow a fixed list of section-heading patterns (e.g. "Cone \d", "Recipe", "Glaze") in addition to query-derived words, so documents that don't use the exact query wording still produce sections.
- **Option C:** If no query word ever matches any span, consider a fallback "everything on the page" or "every block" as one chunk so the index is never empty for a non-empty PDF.

---

### 3.3 Avoid dropping single-heading sections and trailing recipe lines

- **Option A:** When "unrelated" text is seen, always flush `current_group` if non-empty (e.g. `if current_group: chunks.append(...)`), even when `len(current_group) == 1`, so a heading-only section still becomes one chunk.
- **Option B:** Don't immediately set `inside_section = False` on first non-matching line; only flush and exit the section after N consecutive non-recipe lines, or after a line that looks like a new heading (e.g. matches heading pattern again).
- **Option C:** Treat "label" lines (e.g. lines matching `Ingredients:`, `Instructions:`, or lines with no digits but short) as part of the section (append to `current_group`) instead of as section-ending "unrelated" text.

---

### 3.4 Relax fallback heading pattern

- Change the fallback from `r"glaze$"` to something like `r"\b(glaze|recipe|cone)\b"` so any of these words (with word boundaries) can start a section when the query doesn't supply longer words.

---

### 3.5 Add diagnostics

- Log or print: number of pages, number of spans per page (with text), whether the heading pattern matched on each page, and how many chunks were appended (and from which pages). Optionally, write a small "debug" mode that prints the first few span texts and whether they matched the heading pattern, so users can see why no sections were detected.

---

## 4. Summary Table

| Flaw | Cause of empty chunks | Proposed direction |
|------|------------------------|--------------------|
| No text in spans | Image-only or unusual PDF | OCR path or clear "no text" message |
| Heading never matches | Query words not in PDF; strict `glaze$` fallback | Broaden fallback; fixed heading keywords; or fallback to page-level chunks |
| Single-heading section dropped | One non-recipe line after heading clears section and skips following recipe lines | Always flush non-empty `current_group`; or allow label lines; or require N non-matching lines before closing section |
| `glaze$` too strict | Headings like "Glaze Recipe" don't match | Use word-boundary or multi-keyword fallback |
| No diagnostics | Hard to debug | Log spans/page, heading matches, and chunk counts |

Implementing 3.2 (broader heading detection), 3.3 (flush single-heading sections and/or allow label lines), and 3.5 (diagnostics) will most directly address the "no chunks" outcome while keeping the current design; 3.1 and 3.4 improve robustness for real-world PDFs and short queries.

---

## 5. Post-change: No results / no partial response (image-heavy PDFs)

After adding OCR + multimodal and tuning (expansion context, TOP_K_RERANK, MIN_SIM_THRESHOLD), the pipeline can produce **no results** or **no partial response** when the PDF has **many images**. Below are flaws in chunking/parsing and retrieval for that case, and proposed solutions.

---

### 5.1 Uncaught exceptions in the image loop

**What happens:** For each image we do:

1. `fitz.Pixmap` / `tobytes` / `Image.open` (inside try/except → skip on failure).
2. `pytesseract.image_to_string(image)` — **not** in try/except.
3. `clean_ocr_text(raw_ocr)` — **not** in try/except.
4. `page.get_image_bbox(img)` (inside try/except → skip on failure).

If `pytesseract.image_to_string` raises (Tesseract missing, image too large, unsupported format, timeout) or `clean_ocr_text` raises (e.g. non-string input in edge cases), the **entire script crashes**. So the user sees no results because the run never reaches the final answer.

**Why it’s a flaw:** One bad or oversized image can stop ingestion for the whole PDF. PDFs with many images are more likely to include at least one such image.

**Proposed solutions:**

- Wrap the full per-image body (from Pixmap through `combined`) in a single `try/except`: on exception, log or increment a skipped count and `continue`, so one bad image does not abort the loop.
- Optionally get `img_bbox` **first** and only run OCR if bbox is valid, to avoid wasting OCR on images that would be dropped later.

---

### 5.2 Cost and fragility when there are many images

**What happens:** The code runs OCR on **every** image on every page. For a PDF with many images:

- Runtime grows linearly with image count (each OCR call is costly).
- Memory can spike (Pixmap + PIL Image + PNG bytes per image); very large images can cause OOM or slow runs.
- Most images may be non-informative (decorative, logos, photos). Their OCR adds noise and fills the index with low-value figure_text chunks.

**Why it’s a flaw:** Image-heavy PDFs become slow and brittle, and the index can be dominated by low-signal figure_text, diluting retrieval for actual recipe/section text.

**Proposed solutions:**

- **Limit images per page:** e.g. process at most N images per page (e.g. 3–5), choosing by size (prefer medium) or by position (e.g. first N), so ingestion stays bounded.
- **Skip by size:** skip images with width or height above a threshold (e.g. 2000 px) to avoid huge Pixmaps and timeouts.
- **Optional: prefer “likely informative” images** (e.g. with a caption block nearby, or aspect ratio suggestive of a table/diagram) and process those first.

---

### 5.3 Image-heavy pages yield little or no section text

**What happens:** In PDFs that are mostly figures and captions:

- `rawdict` still returns spans, but they are often **short captions** (“Figure 3”, “See table below”, “Temmoku sample”). The **heading pattern** (query-derived or `glaze$`) may not match these captions, so `inside_section` is never set and **no section chunks** are produced for that page.
- Section logic also requires **cone/recipe lines** after a heading. If the only text is caption-style, we never build section chunks.
- So for image-heavy pages we rely entirely on **figure_text** from OCR. If `clean_ocr_text` is strict (min length, alphanumeric ratio, total length &lt; 10), many images yield **no** figure_text chunk. Result: **whole pages can contribute zero chunks** to the index.

**Why it’s a flaw:** The index can end up with few or no chunks that contain the actual recipe text, so retrieval returns nothing useful and the model has “no (or no partial) response”.

**Proposed solutions:**

- **Page-level fallback:** If after semantic grouping a page has **zero section chunks**, build one **page-level chunk** from the page’s text (e.g. `"\n".join(tb["text"] for tb in text_blocks)` or from rawdict blocks) and append it with `type: "section"` (or `"page"`). That way every page with any text contributes at least one chunk, so retrieval has something to work with.
- Combine with 3.2/3.3 (broader heading, flush single-heading sections) so more caption/nearby text can form section-like chunks when they match recipe/cone patterns.

---

### 5.4 Distance threshold and “no results” after filtering

**What happens:** The pipeline keeps documents where `info["dist"] < MIN_SIM_THRESHOLD` (e.g. 0.45). ChromaDB’s **default** is **L2 (squared) distance**; the collection is created with `get_or_create_collection` and no `metadata={"hnsw:space": "cosine"}`, so existing collections may use L2. For L2, typical distances are often **&gt; 0.5** even for relevant docs. Then **all** docs can have `dist >= 0.45`, so `filtered_docs` becomes **empty**. The code then falls back to **all** `unique` docs; if `unique` is large and mostly figure_text, reranking can still surface poor chunks, and the model may answer “no information” or give a useless reply. In the worst case (e.g. empty or tiny `unique`), **context** for the final answer is empty or negligible → **no or no partial response**.

**Why it’s a flaw:** The threshold is interpreted as if it were cosine distance; with L2 it can over-filter and leave retrieval/rerank with bad or empty context.

**Proposed solutions:**

- **Do not rely on a fixed distance value:** either **remove** the distance filter and always pass all `unique` to the reranker, or use a **percentile** (e.g. keep docs with dist in the bottom 50% per query) so behavior adapts to the actual distance distribution.
- **Or** create the collection with `metadata={"hnsw:space": "cosine"}` so distances are in [0, 2] and a threshold like 0.45 has a consistent meaning (requires re-indexing).
- If keeping a threshold: make it **configurable** and document whether it is for L2 or cosine; for L2, use a **larger** value (e.g. 1.0 or 2.0) so that relevant docs are not all filtered out.

---

### 5.5 Empty or useless final answer

**What happens:** Even when `context` is non-empty, the model can return an empty string (API/error) or a generic “I don’t have that information” if the retrieved chunks don’t contain the recipe. The user then sees “no results” or “no partial response”.

**Why it’s a flaw:** There is no handling for empty `final_answer` and no fallback (e.g. “No relevant excerpts found” or retry with a simpler prompt), so the output looks broken.

**Proposed solutions:**

- After `generate_final_answer`, if `not final_answer.strip()`: print a clear fallback message (e.g. “No answer could be generated from the retrieved excerpts.”) and optionally log `len(context)`, `len(top_docs)`.
- Optionally retry once with a shorter or more permissive prompt (e.g. “Summarize the following excerpts in one paragraph.”) to get at least a partial response when the main prompt yields nothing.

---

### 5.6 Summary: image-heavy PDFs and no results

| Flaw | Cause | Proposed direction |
|------|--------|---------------------|
| Uncaught OCR/clean exceptions | One bad image crashes the whole run | try/except around full per-image body; optionally get bbox first |
| Many images | Slow, memory-heavy, noisy index | Limit images per page; skip very large images |
| No section chunks on image-heavy pages | Heading/recipe logic doesn’t match captions; strict OCR cleaning | Page-level fallback chunk when section count is 0; relax heading/OCR rules |
| Distance threshold (L2 vs 0.45) | Over-filter → empty or bad context | Use percentile or no filter; or use cosine + re-index; or raise L2 threshold |
| Empty final answer | Model or API returns nothing | Check for empty answer; print fallback message; optional retry |

Implementing 5.1 (robust image loop), 5.3 (page-level fallback), and 5.4 (threshold/percentile or cosine) will most directly address “no results” after the new changes; 5.2 and 5.5 improve robustness and UX.

---

## 6. Why “No documents to add” when dropping images and retaining text (PDFs with images)

When the pipeline is configured to **drop images** and use **only text from the PDF** (no OCR, no figure_text from images), the error **“No documents to add: parse_pdf_to_chunks produced no chunks”** often appears for PDFs that **contain many images**. This section explains the underlying parsing flaw.

---

### 6.1 What “drop images, retain text” actually uses

- **Dropping images** means: we do **not** call `page.get_images()`, do **not** run OCR on images, and do **not** create any chunks from image content (caption, nearby text, or OCR).
- **Retaining text** means: we use **only** the PDF **text layer** via `page.get_text("rawdict")`. That is the embedded text that PDF readers can select and copy — not pixels inside images.

So the parser’s only source of content is the rawdict text layer. Anything that exists only inside images is invisible to this path.

---

### 6.2 Why PDFs with many images often have little or no usable text layer

In many PDFs that are **image-heavy** (recipes, brochures, scanned docs, designed layouts):

1. **Content lives in the images**  
   The actual recipe text (ingredients, amounts, cone, instructions) is often **rendered as part of the image** (e.g. a photo of a recipe card, or a designed page where text was exported as pixels). In that case there is **no** corresponding text in the PDF text layer for that content.

2. **Text layer is missing or minimal**  
   - Scanned PDFs often have **no** text layer unless someone ran OCR and added it.
   - Some PDFs have a text layer that only includes **short labels** (e.g. “Figure 3”, “See below”, page numbers). The body content is only in the images.
   - So `raw["blocks"]` can be **empty**, or contain only a few blocks with little text.

3. **rawdict only sees the text layer**  
   `get_text("rawdict")` returns **only** text that the PDF stores as text (with position/font info). It does **not** see text that is drawn inside images. So for image-heavy PDFs, rawdict often yields **no spans** or **very few spans** (e.g. captions only).

Result: after dropping images, the **only** source of text is the text layer, which for these PDFs is empty or minimal → we extract **no or almost no spans** → no chunks.

---

### 6.3 How the semantic grouping makes it worse

Even when the text layer has *some* text, the current logic can still produce **zero chunks**:

- A **section** is only started when a span matches the **heading pattern** (query-derived words or `glaze$`).
- Only **cone lines** or **recipe lines** (e.g. `[A-Za-z].*\d`) are then added to the section; any other line (e.g. “Ingredients:”) can **close** the section and, if the section has only one line, **discard** it (see §2.3).
- So we only get chunks when:
  1. At least one span matches the heading pattern, and  
  2. We then see cone/recipe lines (or we flush a multi-line group).

If the text layer has only captions like “Temmoku”, “Cone 6”, “Recipe” in **separate** blocks or in wording that doesn’t match the pattern (e.g. “Cone 6” without “glaze”), we may never start a section, or we may start one and then immediately discard it. So **even a non-empty text layer can yield zero chunks** for an image-heavy PDF where the “real” content is in the images.

---

### 6.4 Root cause in one sentence

**When we drop images and retain only text, we rely 100% on the PDF text layer; for many PDFs that contain images, the meaningful content is inside the images and the text layer is empty or minimal and/or doesn’t match the strict heading/recipe grouping, so the parser correctly has nothing to chunk and returns zero chunks.**

---

### 6.5 What to do about it

| Goal | Option |
|------|--------|
| Keep “text only” but avoid empty index | Add a **page-level fallback**: if a page yields **zero section chunks** from rawdict/semantic grouping, build **one chunk from all text on that page** (e.g. `get_text("blocks")` or rawdict blocks concatenated). Then every page with *any* text layer content contributes at least one chunk (§3.5, §5.3). |
| Use content that is inside images | **Re-enable image processing** (OCR + caption + nearby text) for figure_text chunks, and keep section chunks from the text layer. Query results can then come from both text and image-derived chunks. |
| Prefer text but allow images when needed | Use a **hybrid**: try text-only first; if `parse_pdf_to_chunks` returns zero chunks, run a second pass that adds page-level text chunks and/or image OCR chunks so the index is never empty when the PDF has any content. |
| Relax grouping so more text becomes chunks | Broaden heading pattern (e.g. fixed keywords like “recipe”, “cone”, “glaze” with word boundaries), always flush non-empty `current_group` (including single-line sections), and optionally treat label lines as part of the section (§3.2, §3.3). |

Implementing the **page-level fallback** (and optionally relaxed grouping) is the smallest change that prevents “No documents to add” for PDFs whose text layer exists but doesn’t match the current section logic. For PDFs where the text layer is truly empty, the only way to get content is to use images again (OCR) or to use a different PDF that has a proper text layer.
