# Import PyMuPDF
import fitz

# File path you want to extract images from
file = "data/recipes.pdf"

# Open the file
pdf_file = fitz.open(file)

# Iterate over PDF pages
for page_index in range(len(pdf_file)):
    # Get the page itself
    page = pdf_file[page_index]

    # Get the image list for the page
    image_list = page.get_images(full=True)

    # Printing the number of images found on this page
    if image_list:
        print(f"[+] Found a total of {len(image_list)} images in page {page_index + 1}")
    else:
        print("[!] No images found on page", page_index + 1)

    # Extract images from the page
    for image_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = pdf_file.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        # Save the image to a file
        image_filename = f"data/images/page_{page_index + 1}_image_{image_index}.{image_ext}"
        with open(image_filename, "wb") as img_file:
            img_file.write(image_bytes)

# Close the PDF document
pdf_file.close()