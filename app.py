import streamlit as st
import os
from exp_query_final_cond import rag_pipeline

def init_state():
    defaults = {
        "open_api_key": None, 
        "pdf_path": None,
        "query": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

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
    st.title("Enter OpenAI key")

    with st.form(key='my_form', clear_on_submit=True):
      user_input = st.text_input("Key")
      submit_button = st.form_submit_button(label='Submit')

      if submit_button:
        # save key to session state
        st.session_state.open_api_key = user_input
        st.write("OpenAI API key saved")

    st.title("Doc Upload")
    st.session_state.pdf_path = st.file_uploader("", type=["pdf"])
    print("pdf_path: ", st.session_state.pdf_path)

# search button
if st.button("Query") and st.session_state.open_api_key:
    st.session_state.query = user_query
    if st.session_state.query and st.session_state.pdf_path:
        full_path = os.path.join(UPLOAD_DIR, st.session_state.pdf_path.name)
        with open(full_path, "wb") as f:
          f.write(st.session_state.pdf_path.getvalue())

        result = rag_pipeline(st.session_state.query, pdf_path=full_path, openai_api_key=st.session_state.open_api_key)
        st.write(result)
    else:
        st.write("Please upload a PDF file")
elif not st.session_state.open_api_key:
    st.write("Please enter your OpenAi key")

