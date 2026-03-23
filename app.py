import streamlit as st
from rag_pipeline import load_vectorstore, generate_answer

st.title("📚 RAG Q&A System")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Processing (first run may take ~1-2 min)..."):
        vectorstore = load_vectorstore()   # ✅ ONLY here
        answer = generate_answer(query, vectorstore)
        st.write(answer)
