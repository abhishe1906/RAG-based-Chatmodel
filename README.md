# 📚 Deep Learning RAG Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions from a Deep Learning textbook and receive accurate, context-grounded answers.

---

## 🚀 Live Demo

👉 [Add your Streamlit App Link here]

---

## 🧠 Problem Statement

Large Language Models (LLMs) often generate hallucinated or generic responses when asked domain-specific questions.

This project solves that by:

* Retrieving **relevant context from a textbook**
* Feeding it into an LLM
* Generating **accurate, grounded answers**

---

## ⚙️ System Architecture

```
User Query
    ↓
Retriever (FAISS)
    ↓
Relevant Chunks
    ↓
LLM (Groq - LLaMA 3.1)
    ↓
Final Answer
```

---

## 🏗️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **LLM API**: Groq (`llama-3.1-8b-instant`)
* **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector DB**: FAISS
* **Document Processing**: PyMuPDF
* **Cloud Storage**: Google Drive

---

## 🔄 Workflow

1. 📥 Download PDF from Google Drive
2. 🧹 Clean and preprocess text
3. ✂️ Split into chunks
4. 🔢 Generate embeddings
5. 📦 Store in FAISS index
6. 🔍 Retrieve top-k relevant chunks
7. 🤖 Generate answer using LLM

---

## ✨ Features

* 🔍 Context-aware Q&A from textbook
* ⚡ Fast inference using Groq
* 📄 Automatic PDF ingestion
* 🧠 Retrieval-based accuracy (reduces hallucination)
* ☁️ Deployed on Streamlit Cloud

---

## 📸 Sample Queries

* *What is gradient descent?*
* *Explain backpropagation*
* *What are activation functions?*

---

## ⚠️ Limitations

* First query may be slow (index creation)
* Limited to single document
* Depends on chunking quality

---

## 🔧 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API Key

Create `.env` file:

```
GROQ_API_KEY=your_api_key
```

### 4. Run Application

```bash
streamlit run app.py
```

---

## ☁️ Deployment

Deployed using **Streamlit Cloud**:

* Add `GROQ_API_KEY` in **Secrets**
* Connect GitHub repo
* Deploy `app.py`

---

## 📈 Future Improvements

* 📚 Multi-document support
* 💬 Conversational memory (chat history)
* 🔎 Hybrid search (BM25 + vector)
* ⚡ Precomputed FAISS index (faster startup)
* 🧪 Evaluation using RAGAS

---

## 👨‍💻 Author

**Abhishek Agrawal**
M.Tech (Signal Processing & ML), NIT Jalandhar

---

## ⭐ Acknowledgements

* Groq for fast LLM inference
* LangChain ecosystem
* HuggingFace for embeddings

---

## 📌 Key Takeaway

> This project demonstrates how combining retrieval with LLMs significantly improves answer accuracy and reliability in domain-specific applications.

---

