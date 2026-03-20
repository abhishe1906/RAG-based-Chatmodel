#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fitz 
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# In[2]:


file_path = "DL_book_Yoshua.pdf"

documents = []

with fitz.open(file_path) as pdf:
    for i, page in enumerate(pdf):
        text = page.get_text()
        
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "page": i
                }
            )
        )

print(f"Total pages loaded: {len(documents)}")


# In[3]:


def clean_text(text):
    text = text.replace("\n", " ") 
    # Fix spacing issues like "DeepLearning"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text) 
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
# Apply cleaning
for doc in documents:
    doc.page_content = clean_text(doc.page_content)


# In[4]:


filtered_docs = []

for doc in documents:
    text = doc.page_content.lower()
    # Remove very short pages
    if len(text) < 50:
        continue
    # Remove useless pages
    if any(x in text for x in ["copyright", "isbn"]):
        continue
    filtered_docs.append(doc)

print(f"Filtered pages: {len(filtered_docs)}")


# In[5]:


filtered_docs = []

for doc in documents:
    if doc.metadata["page"] < 14:  # skip title, TOC, etc.
        continue
    filtered_docs.append(doc)


# In[6]:


def is_low_information(text):
    # too many dots or short fragments
    if text.count("...") > 3:
        return True
    if len(text.split()) < 30:
        return True
    return False


# In[8]:


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(filtered_docs)

print(f"Total chunks: {len(chunks)}")


# In[9]:


final_chunks = []
for chunk in chunks:
    if is_low_information(chunk.page_content):
        continue
    final_chunks.append(chunk)


# In[10]:


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)


# In[11]:


print(f"Total chunks: {len(chunks)}")


# In[12]:


for i, chunk in enumerate(chunks):
    chunk.metadata.update({
        "chunk_id": i,
        "type": "content"
    })


# In[13]:


for i in range(3):
    print("TEXT:\n", chunks[i].page_content[:300])
    print("METADATA:\n", chunks[i].metadata)
    print("--------")


# In[14]:


def is_math_table(text):
    symbols = ["∂", "∇", "∑", "∫", "∈"]
    count = sum(text.count(s) for s in symbols)
    
    return count > 10  # heuristic


# In[15]:


def normalize_math(text):
    # Fix H( ) x → H(x)
    text = re.sub(r'([A-Za-z])\(\s*\)\s*([A-Za-z])', r'\1(\2)', text)
    # Fix P( ) a → P(a)
    text = re.sub(r'P\(\s*\)\s*([a-zA-Z])', r'P(\1)', text)
    # Fix spacing around symbols
    text = re.sub(r'\s*([=+\-*/])\s*', r' \1 ', text)
    return text


# In[17]:


for doc in documents:
    text = clean_text(doc.page_content)
    text = normalize_math(text)
    doc.page_content = text


# In[18]:


def is_low_information(text):
    if len(text.split()) < 40:
        return True
    # Too many dots / broken formatting
    if text.count("...") > 2:
        return True   
    return False
final_chunks = []
for chunk in chunks:
    if is_low_information(chunk.page_content):
        continue
    final_chunks.append(chunk)

print(f"Final chunks: {len(final_chunks)}")


# In[19]:


for i in range(5):
    print("TEXT:\n", final_chunks[i].page_content[:300])
    print("METADATA:\n", final_chunks[i].metadata)
    print("--------")


# In[21]:


from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbedding
import numpy as np

vector_db = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
# In[22]:


model = SentenceTransformer("all-MiniLM-L6-v2")


# In[23]:


texts = [doc.page_content for doc in final_chunks]
embeddings = model.encode(texts, show_progress_bar=True)


# In[25]:


import faiss
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# In[26]:


docstore = final_chunks


# In[27]:


def retrieve(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = []
    for idx in indices[0]:
        results.append(docstore[idx])
    return results


# In[30]:


query = "What is the origin of Deep Learning"
results = retrieve(query, k=10)
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content[:300])
    print(doc.metadata)


# In[32]:


import ollama
response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": "What is gradient descent?"}]
)
print(response["message"]["content"])


# In[33]:


def generate_answer(query):
    docs = retrieve(query, k=3)
    
    context = "\n\n".join([doc.page_content for doc in docs])  
    prompt = f"""
You are a helpful AI tutor.
Answer ONLY using the context below.
Be friendly in your tone and prioritize accuracy of the response.
If the answer is not present, say "Not found in provided material".
Context:
{context}
Question:
{query}
"""
    
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# In[1]:


import streamlit as st


# In[2]:


st.title("📚 RAG Q&A System")
query = st.text_input("Enter your question:")
if st.button("Ask"):
    if query:
        answer = generate_answer(query)
        st.write("### Answer:")
        st.write(answer)


def generate_answer(query):
    docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return context 
    
def generate_answer(query):
    docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
Answer using only the context below.

Context:
{context}

Question:
{query}
"""
    
    import ollama
    
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]
from groq import Groq

client = Groq(api_key="your_groq_api_key")

def generate_answer(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }]
    )
    
    return response.choices[0].message.content





