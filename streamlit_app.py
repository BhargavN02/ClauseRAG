import streamlit as st
import os
import tempfile
import faiss
import numpy as np

from docx import Document
import fitz  # PyMuPDF for PDFs

from sentence_transformers import SentenceTransformer
from transformers import pipeline

##########################################
# Initialize models at app start
##########################################
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

GEN_MODEL_NAME = "google/flan-t5-base"  # or "meta-llama/Llama-2-7b-hf", etc.
generator = pipeline(task="text2text-generation", model=GEN_MODEL_NAME)

##########################################
# Utility functions
##########################################
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text() + " "
    return text

def extract_text_from_docx(file_path: str) -> str:
    text = ""
    doc = Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_faiss_index(chunks):
    """
    Given a list of text chunks, embeds them and builds a FAISS index in memory.
    Returns (faiss_index, chunk_embeddings, chunk_texts).
    """
    # 1. Embed the chunks
    chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)
    chunk_embeddings = chunk_embeddings.astype("float32")

    # 2. Create FAISS index
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_embeddings)

    return index, chunk_embeddings, chunks

def retrieve_top_k(query, faiss_index, chunk_texts, top_k=3):
    """
    Embeds the query, searches FAISS, returns top_k chunk texts.
    """
    query_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = faiss_index.search(query_emb, top_k)
    results = []
    for rank in range(top_k):
        idx = indices[0][rank]
        dist = distances[0][rank]
        results.append(chunk_texts[idx])
    return results

def generate_answer(query, retrieved_chunks):
    """
    Builds a prompt with the retrieved chunk texts and uses the local model to generate an answer.
    """
    context_text = "\n\n".join([f"Context chunk:\n{c}" for c in retrieved_chunks])
    prompt = (
        f"You are an AI assistant. Use the following context to answer the question.\n\n"
        f"{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer concisely:"
    )

    output = generator(prompt, max_length=256, do_sample=False)
    return output[0]["generated_text"]

##########################################
# Streamlit App
##########################################
def main():
    st.title("Instant Legal Clause Finder")

    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])
    if uploaded_file is not None:
        # 2. Read the file to a temp location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_filepath = tmp_file.name

        # 3. Extract text
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".pdf":
            text = extract_text_from_pdf(temp_filepath)
        else:
            # docx
            text = extract_text_from_docx(temp_filepath)

        # Clean up whitespace
        text = " ".join(text.split())

        # 4. Chunk the text
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        # 5. Build a FAISS index in memory
        faiss_index, chunk_embeddings, chunk_texts = create_faiss_index(chunks)

        st.success("File processed successfully! You can now ask questions below.")

        # 6. Display a text input for queries
        user_query = st.text_input("Enter your question about this document:")
        if user_query:
            with st.spinner("Generating answer..."):
                # 7. Retrieve top-k chunks and generate answer
                retrieved = retrieve_top_k(user_query, faiss_index, chunk_texts, top_k=3)
                answer = generate_answer(user_query, retrieved)
            st.write("**Answer:**", answer)
    else:
        st.info("Please upload a PDF or DOCX file to get started.")

if __name__ == "__main__":
    main()
