# Why “No documents to add”: Parser/chunking logic not retrieving text from the PDF

**Script:** `exp_query_final_cond.py`  
**Error:** `No documents to add: parse_pdf_to_chunks produced no chunks (empty PDF, no matching sections, or no text in spans).`  
**Context:** The PDF has many images and little text. There is no context to gather from images; the failure is due to the **parser not retrieving text** that exists in the PDF.

This document analyzes the **parser and chunking logic** and reports why text in the PDF is not turned into chunks.

---

## 1. Pipeline overview

1. **`parse_pdf_to_chunks(pdf_path, user_query)`** — Reads the PDF, extracts text via rawdict, groups it by heading/cone/recipe rules, returns a list of groups (each with `text` and `metadata`).  
2. **Index build** — For each group, `chunk_text(group["text"], page)` splits the text into smaller chunks; these are added to ChromaDB as `docs` and `metas`.  
3. If `parse_pdf_to_chunks` returns **no groups**, `docs` stays empty and the script exits with “No documents to add.”

So the error means **the parser produced zero groups**. The cause is in how text is extracted and which text is allowed to become a chunk.

---

## 2. Parser logic: extraction (rawdict)

**Code path:** `page.get_text("rawdict")` → iterate `raw["blocks"]` → for each block, require `"lines"` in block → for each line, iterate `line["spans"]` → keep span only if `(span.get("text") or "").strip()` is non-empty.

### 2.1 Blocks without `"lines"` are skipped entirely

- Only blocks that have a `"lines"` key are processed. Any block that does **not** have `"lines"` is skipped; its content is **never read**.
- In PyMuPDF’s rawdict, **image blocks** typically have no `"lines"` (they are not text). So skipping them is correct. But some PDFs store or expose text in structures that rawdict represents as blocks **without** a standard `"lines"` structure (e.g. annotations, form fields, or non-standard layouts). In those cases, that text is **never extracted** by this loop.
- **Finding:** Text that lives in blocks without a `"lines"` key is not retrieved.

### 2.2 Spans without text are skipped

- Spans with no `"text"` key or empty text are skipped (`if not text: continue`). That is correct. The only risk is PDFs where the key has a different name; the code does not try alternative keys.
- **Finding:** Only spans with non-empty `span["text"]` (or equivalent) contribute. Other span content is not retrieved.

### 2.3 No other text source

- The parser uses **only** rawdict. It does not use `get_text("blocks")`, `get_text("text")`, or any other extraction method. So if rawdict omits or structures text in a way this loop doesn’t handle, that text is never retrieved.
- **Finding:** Any text not present in rawdict blocks that have `"lines"` and spans with non-empty text is invisible to the parser.

---

## 3. Parser logic: semantic grouping (why extracted text still doesn’t become chunks)

Assume the extraction step above has produced a non-empty list `spans` of `{ "text", "bbox", "y" }` for the page. Chunks are created **only** by the following rules.

### 3.1 A section only starts when a span matches the heading pattern

- `heading_pattern` is built from the **user query**: words with length &gt; 2, joined with `|`, case-insensitive. Example: “What is the recipe for temmoku glaze?” → `(what|recipe|for|temmoku|glaze)`.
- If the query has no word longer than 2 characters, the fallback is `glaze$` (line must **end** with “glaze”).
- A span starts a new section **only if** `heading_pattern.search(t)` is true. So at least one span on the page must contain one of those words (or end with “glaze” in the fallback case).
- **Finding:** Any page where **no** span matches the heading pattern **never starts a section**. All spans on that page are then considered only as “inside section” or “not inside section.” Since we never set `inside_section = True`, we **never add any span** to `current_group`. So **all extracted text on that page is discarded** and produces zero chunks.

So even if the PDF has retrievable text (e.g. “Temmoku”, “Cone 6”, “Feldspar 25”), if **no** span contains a query word (or “glaze” at end of line), the parser **does not retrieve that text into any chunk**.

### 3.2 Only cone and recipe lines are added to the section

- While `inside_section` is True, a span is appended to `current_group` **only if** it matches:
  - `cone_pattern`: `r"cone\s*\d"`, or  
  - `recipe_line_pattern`: `r"[A-Za-z].*\d"` (at least one letter and at least one digit).
- Any other line (e.g. “Ingredients:”, “Instructions:”, “Temmoku glaze”, “Notes”) does **not** match, so the code treats it as **unrelated** and exits the section.
- **Finding:** Text that does not match cone or recipe pattern is never added to a section. It is only “used” to close the current section (and possibly discard it; see below).

### 3.3 Single-line sections are discarded

- When we see an “unrelated” line (doesn’t match cone/recipe), we do:
  - If `len(current_group) > 1`: append a chunk for `current_group`.
  - Then set `current_group = []` and `inside_section = False`.
- If `len(current_group) == 1` (only the heading line), we **do not** append any chunk; we only clear the group. So the heading line is **never** written to the chunk list.
- **Finding:** A section that consists of only the heading (e.g. one span “Temmoku glaze”) is **discarded** as soon as the next span doesn’t match cone/recipe. So that text is **not retrieved** into any chunk.

### 3.4 End-of-page flush keeps single-line groups

- At the end of the page we do: `if current_group: chunks.append(...)`. So a group that has only one line **is** output if we reach end of page without seeing an “unrelated” line. So the only way a single-line section is lost is when an unrelated line appears **before** end of page (triggering the discard in 3.3).

### 3.5 No fallback for “any text on the page”

- There is **no** rule that says: if this page has any spans, emit at least one chunk (e.g. “all text on the page” or “every block as one chunk”). So if the heading pattern never matches, or every section is discarded as single-line, the page contributes **zero chunks** even when it had extractable text.
- **Finding:** The parser does not have a page-level or block-level fallback; it only outputs chunks that pass the strict heading + cone/recipe grouping.

---

## 4. Chunking (downstream of parser)

- `chunk_text(text, page_number)` only runs on the **group text** that the parser already output. If the parser outputs no groups, `chunk_text` is never called for that page, so there are no documents to add.
- So the “no chunks” error is **not** caused by `chunk_text` (e.g. splitting empty string); it is caused by **parse_pdf_to_chunks** returning an empty list. The chunking step is irrelevant when the parser retrieves no text into groups.

---

## 5. Findings summary: why the parser does not retrieve text

| # | Finding | Effect on “no chunks” |
|---|--------|------------------------|
| 1 | Blocks without `"lines"` are skipped | Text in such blocks is never extracted. |
| 2 | Only rawdict is used | Text not exposed via rawdict (or in non-standard structure) is never retrieved. |
| 3 | Section starts only on heading match | If no span matches the query-derived pattern (or `glaze$`), no section ever starts; all spans on the page are discarded. |
| 4 | Only cone/recipe lines added to section | All other text (labels, prose, titles without numbers) is never added to any group. |
| 5 | Single-line sections discarded on “unrelated” line | When the next line doesn’t match cone/recipe, a section with only the heading is dropped; that heading text is not retrieved. |
| 6 | No page-level or block-level fallback | Pages with text that doesn’t pass the strict rules contribute zero chunks. |

**Root cause:** The parser is designed to output only “section” chunks that start with a heading (query words or “glaze”) and contain only cone/recipe lines. It **does not retrieve** (into any chunk): text in blocks without `"lines"`, text that never matches the heading pattern, text that doesn’t match cone/recipe, or single-line sections when the next line is “unrelated.” For a PDF with many images and little text, the little text that exists is often captions, labels, or short lines that fail these rules, so the parser correctly returns no groups and the script reports “No documents to add.”

---

## 6. Recommended changes (so the parser retrieves text)

- **Page-level fallback:** For each page, if after the current grouping logic the page has **zero** chunks, build **one chunk** from all extracted spans on that page (e.g. `"\n".join(sp["text"] for sp in spans)`). Then any page with at least one span will contribute at least one chunk.
- **Always flush non-empty section:** When an “unrelated” line is seen, always append `current_group` to chunks if it is non-empty (even when `len(current_group) == 1`), so heading-only sections are still retrieved.
- **Broaden what starts a section or what gets added:** e.g. fixed keywords like “recipe”, “glaze”, “cone” (with word boundaries) in addition to query words; or allow short label lines (“Ingredients:”, etc.) to be added to the section instead of closing it.
- **Optional: use another extraction path** for pages where rawdict yields no `"lines"` blocks (e.g. `get_text("blocks")` or `get_text("text")`) and merge that text into a page-level chunk so no available text is ignored.

These changes address the parser/chunking logic so that text that exists in the PDF is actually retrieved and turned into chunks, without requiring any context from images.
