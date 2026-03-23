import streamlit as st
from rag_pipeline import load_vectorstore, generate_answer

@st.cache_resource
def get_vectorstore():
    return load_vectorstore()

st.title("📚 RAG Q&A")

query = st.text_input("Ask something")

if st.button("Ask"):
    with st.spinner("Processing..."):
        vectorstore = get_vectorstore()
        answer = generate_answer(query, vectorstore)
        st.write(answer)
