# app.py - Milestone 2 Local-only RAG PDF Chatbot

import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Simple RAG PDF Chatbot", layout="wide")
st.title("📄 Simple RAG PDF Chatbot (Local-only)")

# 1️⃣ Load sentence-transformers model
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# 2️⃣ Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    
    # 3️⃣ Debug: check first 500 characters
    st.write("Preview of PDF text:", text[:500])

    # 4️⃣ Split text into chunks
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # 5️⃣ Debug: check number of chunks
    st.write("Number of chunks created:", len(chunks))

    # 6️⃣ Create embeddings for all chunks
    @st.cache_resource(show_spinner=True)
    def create_embeddings(chunks_list):
        return model.encode(chunks_list)
    
    embeddings = create_embeddings(chunks)

    # 7️⃣ User question input
    question = st.text_input("Ask question from PDF")
    if question:
        question_emb = model.encode([question])[0]
        similarities = cosine_similarity([question_emb], embeddings)[0]
        best_idx = np.argmax(similarities)
        answer = chunks[best_idx]
        st.markdown("**Answer (from PDF):**")
        st.write(answer)