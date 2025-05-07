# app.py
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import os
from utils import save_uploaded_file, extract_text

import streamlit as st
from transformers import pipeline
from utils import save_uploaded_file, extract_text

# Streamlit UI
st.set_page_config(page_title="RAGify: AI-Powered Summarizer")
st.title("📄 RAGify: Document Summarizer using Transformers")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    raw_text = extract_text(file_path)
    st.success("✅ File processed successfully")

    # Load summarization pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    if st.button("Summarize File"):
        # Split long text if needed
        max_chunk_len = 1000
        chunks = [raw_text[i:i+max_chunk_len] for i in range(0, len(raw_text), max_chunk_len)]
        summarized_chunks = []

        for chunk in chunks:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            summarized_chunks.append(summary)

        full_summary = "\n\n".join(summarized_chunks)
        st.subheader("📌 Summary:")
        st.write(full_summary)
