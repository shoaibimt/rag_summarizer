# utils.py
import os
import tempfile
from langchain.document_loaders import TextLoader, PyPDFLoader

def save_uploaded_file(uploaded_file):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_text(file_path):
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)
