#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from groq import Groq
import gdown

def download_pdf():
    file_id = "1giKrO3sTCguTiKKMZuTGdgVHPpHm5TJW"
    output_path = "data/book.pdf"

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    return output_path
def load_vectorstore():
    PDF_PATH = download_pdf()
    
    if not os.path.exists(FAISS_PATH):
        return create_vectorstore(PDF_PATH)
FAISS_PATH = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not set")

client = Groq(api_key=api_key)

def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid(text):
    if len(text.split()) < 40:
        return False
    if text.count("...") > 2:
        return False
    return True
def load_documents(pdf_path):
    documents = []

    with fitz.open(pdf_path) as pdf:
        for i, page in enumerate(pdf):
            if i < 14:
                continue

            text = clean_text(page.get_text())

            if len(text) < 50:
                continue

            documents.append(Document(
                page_content=text,
                metadata={"page": i}
            ))

    return documents
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)
    return [c for c in chunks if is_valid(c.page_content)]
def create_vectorstore(pdf_path):
    docs = load_documents(pdf_path)
    chunks = split_documents(docs)

    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    vectorstore = FAISS.from_documents(
        chunks,
        embedding_model
    )

    vectorstore.save_local(FAISS_PATH)

    return vectorstore

def load_vectorstore():
    pdf_path = download_pdf()
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    if not os.path.exists(FAISS_PATH):
        return create_vectorstore(pdf_path)

    return FAISS.load_local(
        FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
def generate_answer(query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a precise AI tutor.
Answer ONLY from the context below.
If not found, say "Not found in provided material".

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content




