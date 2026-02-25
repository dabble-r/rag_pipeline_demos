import streamlit as st
import os
from exp_query_final_cond import rag_pipeline

# central page
st.title("Glaze Recipe RAG")

# upload PDF directory
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# user query
user_query = st.text_input("Enter your query...")
print("user_query: ", user_query)

# sidebar
with st.sidebar:
    st.title("Upload PDF")
    pdf_path = st.file_uploader("", type=["pdf"])
    print("pdf_path: ", pdf_path)

# search button
if st.button("Search"):
    if pdf_path:
        full_path = os.path.join(UPLOAD_DIR, pdf_path.name)
        with open(full_path, "wb") as f:
          f.write(pdf_path.getvalue())

        result = rag_pipeline(user_query, pdf_path=full_path)
        st.write(result)
    else:
        st.write("Please upload a PDF file")