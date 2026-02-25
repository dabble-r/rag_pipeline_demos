# Installing Tesseract OCR on Ubuntu (step by step)

Tesseract is the OCR engine used by `pytesseract` in this project. The Python package `pytesseract` only talks to the **Tesseract binary**; you must install the binary on your system separately.

## Step 1: Update package lists

```bash
sudo apt update
```

## Step 2: Install Tesseract OCR

```bash
sudo apt install tesseract-ocr
```

This installs the Tesseract engine and default language data (usually English).

## Step 3: Verify the install

```bash
tesseract --version
```

You should see a version line (e.g. `tesseract 4.x.x` or `5.x.x`). If the command is not found, Tesseract is not on your PATH; try logging out and back in, or open a new terminal.

## Step 4 (optional): Install extra language packs

For languages other than the default:

```bash
# List available language packs
apt list tesseract-ocr-*

# Install a specific language, e.g. German and French
sudo apt install tesseract-ocr-deu tesseract-ocr-fra
```

## Step 5: Confirm from Python

With Tesseract installed and on your PATH, the following should run without error:

```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

---

**Summary:** Run `sudo apt update`, then `sudo apt install tesseract-ocr`, then check with `tesseract --version`. After that, `exp_query_multimodal.py` can use OCR on PDF images (or you can keep OCR optional with a try/except and run without installing Tesseract).
