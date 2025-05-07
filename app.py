# app.py
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import os
from utils import save_uploaded_file, extract_text

# Streamlit UI
st.set_page_config(page_title="RAGify: AI-Powered Summarizer")
st.title("📄 RAGify: Document Summarizer using Transformers")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    raw_text = extract_text(file_path)
    st.success("✅ File processed successfully")

    # Chunk the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.create_documents([raw_text])

    # Embed and create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(texts, embeddings)

    # Load transformer summarizer model
    summarizer_pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    llm = HuggingFacePipeline(pipeline=summarizer_pipe)

    # Create RAG chain
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    if st.button("Summarize File"):
        query = "Summarize the entire document briefly."
        try:
            output = rag_chain.invoke({"query": query})
            if output:
                st.subheader("📌 Summary:")
                st.write(output)
            else:
                st.warning("⚠️ No summary was generated. Try another document.")
        except Exception as e:
            st.error(f"❌ Error generating summary: {e}")

        st.caption(f"Document split into {len(texts)} chunks.")

