import fitz

def parse_pdf_to_chunks(pdf_path):
    pdf_document = pdf_path
    doc = fitz.open(pdf_document)
    print("doc page 8: ", doc.get_page_text(8))

    extracted_text = ""

    for page in range(doc.page_count):
        raw = doc.get_page_text(page)

        extracted_text += raw + "\n\n"

    doc.close()
    return extracted_text

if __name__ == "__main__":
    pdf_path = "data/recipes.pdf"
    extracted_text = parse_pdf_to_chunks(pdf_path)
    print("Extracted text:\n", extracted_text)
    word_count = len(extracted_text.split())
    print(f"Total words: {word_count}")