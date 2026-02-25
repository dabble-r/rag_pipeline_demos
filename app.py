import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from exp_query_final_cond import rag_pipeline

def init_state():
    defaults = {
        "open_api_key": None, 
        "pdf_path": None,
        "query": None,
        "client": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# initialize session state
init_state()

def build_page():
  # central page
  st.title("Glaze Recipe RAG")

  # upload PDF directory
  UPLOAD_DIR = "uploaded_pdfs"
  os.makedirs(UPLOAD_DIR, exist_ok=True)

  # user query
  user_query = st.text_input("Enter your query...")
  # print("user_query: ", user_query)

  # sidebar
  with st.sidebar:
      st.title("Enter OpenAI key")

      with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Key")
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
          # save key to session state (strip whitespace; empty input clears key)
          raw = (user_input or "").strip()
          st.session_state.open_api_key = raw if raw else None
          if st.session_state.open_api_key:
            st.write("OpenAI API key saved")
            try:
              client = OpenAI(api_key=st.session_state.open_api_key)
              st.session_state.client = client
              st.write("OpenAI client initialized")
            except Exception as e:
              st.write(f"Key rejected: {e}")
          else:
            st.session_state.open_api_key = None
            st.write("Enter a non-empty API key and submit")

      st.title("Doc Upload")
      st.session_state.pdf_path = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="visible")
      # print("pdf_path: ", st.session_state.pdf_path)

  # search button
  if st.button("Query") and st.session_state.open_api_key:
      key = (st.session_state.open_api_key or "").strip()
      if not key:
          st.write("Please enter your OpenAI key in the sidebar and submit.")
      else:
          st.session_state.query = user_query
          if st.session_state.query and st.session_state.pdf_path:
            full_path = os.path.join(UPLOAD_DIR, st.session_state.pdf_path.name)
            with open(full_path, "wb") as f:
              f.write(st.session_state.pdf_path.getvalue())
            try: 
              result = rag_pipeline(
                  st.session_state.query,
                  full_path,
                  openai_api_key=(st.session_state.open_api_key or "").strip(),
              )
              if result:
                st.write(result)
              else:
                st.write("No results found")
            except Exception as e:
              err_str = str(e)
              if "401" in err_str or "invalid_api_key" in err_str or "Incorrect API key" in err_str:
                st.error(
                    "**Invalid or expired OpenAI API key.**\n\n"
                    "• Get a new key (or check this one) at: https://platform.openai.com/account/api-keys\n"
                    "• Paste the key with no extra spaces or line breaks.\n"
                    "• Enter it in the sidebar and click Submit, then try Query again."
                )
              else:
                st.write(f"Error: {e}")
          else:
            st.write("Please upload a PDF file")
  elif not st.session_state.open_api_key:
      st.write("Please enter your OpenAi key then submit query")

# display main and sidebar
build_page()