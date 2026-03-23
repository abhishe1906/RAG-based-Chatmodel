import streamlit as st
from rag_pipeline import load_vectorstore, generate_answer

st.set_page_config(page_title="RAG Q&A", layout="wide")

st.title("📚 RAG Q&A System")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        vectorstore = load_vectorstore()
        answer = generate_answer(query, vectorstore)

        st.subheader("Answer")
        st.write(answer)
@st.cache_resource
def load_cached_vectorstore():
    from RAG2(1) import load_vectorstore
    return load_vectorstore()
vectorstore = load_cached_vectorstore()
