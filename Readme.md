# RAGify: AI-Powered Document Summarizer

This is a simple Streamlit app that demonstrates a Retrieval-Augmented Generation (RAG) pipeline using:
- LangChain for vector store and RAG
- Transformers for summarization
- FAISS for efficient semantic search

## 🔧 Features
- Upload `.txt` or `.pdf` files
- Auto chunking and embedding
- Semantic search using FAISS
- Summarization using `distilbart-cnn-12-6`

## 🧠 Technologies Used
- Streamlit
- LangChain
- Hugging Face Transformers
- FAISS
- sentence-transformers

## 🚀 How to Run
1. Clone this repo:
```bash
git clone https://github.com/yourusername/ragify.git
cd ragify
```

2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

## 📄 Output
The app displays a summary of the uploaded document, generated using a transformer model on the top retrieved chunks.

---

### 🧑‍💼 Perfect for Interviews
Showcase your skills in:
- Generative AI
- LangChain and RAG
- NLP and document processing

---

Feel free to customize this for your resume or personal projects!
