# src/query.py

import os
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline

##################################
# Setup embedding and generation
##################################
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME   = "google/flan-t5-base"

print(f"Loading SentenceTransformer ({EMBED_MODEL_NAME})...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

print(f"Loading text-generation pipeline with model ({GEN_MODEL_NAME})...")
# For Flan-T5, we use "text2text-generation"
generator = pipeline(
    task="text2text-generation",
    model=GEN_MODEL_NAME
    # If you have an MPS or GPU device, you could do: device=0
)

def load_index_and_metadata():
    """
    Loads the FAISS index (contracts.index) and the metadata (metadata.pkl)
    from the parent directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(os.path.dirname(base_dir), "contracts.index")
    meta_path  = os.path.join(os.path.dirname(base_dir), "metadata.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata_list = pickle.load(f)  # list of dicts
    return index, metadata_list

def embed_text(text: str) -> np.ndarray:
    """
    Embeds the query text using the same SentenceTransformer that was
    used for indexing.
    """
    emb = embedder.encode([text], convert_to_numpy=True)
    return emb[0].astype('float32')

def retrieve_chunks(index, metadata_list, query: str, top_k=3):
    """
    Returns top_k chunks most relevant to the query.
    Each item returned is (chunk_dict, distance).
    """
    query_emb = embed_text(query).reshape(1, -1)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for rank in range(top_k):
        idx = indices[0][rank]
        dist = distances[0][rank]
        chunk_dict = metadata_list[idx]
        results.append((chunk_dict, dist))
    return results

def generate_answer_with_context(query: str, retrieved_chunks):
    """
    Format the context from retrieved chunks and pass it to
    the local LLM for a final answer.
    retrieved_chunks: list of (chunk_dict, distance)
    """
    # Combine text from the retrieved chunks
    context_text = ""
    for chunk_info, dist in retrieved_chunks:
        context_text += (
            f"\n---\nFrom {chunk_info['filename']} (chunk {chunk_info['chunk_id']}):\n"
            f"{chunk_info['text']}"
        )

    # For Flan-T5, we feed an instruction-like prompt
    prompt = (
        f"Given the following legal contract clauses:\n\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Please provide a concise, clear answer. If not sure, say so."
    )

    # Generate a response
    output = generator(prompt, max_length=256, do_sample=False)
    answer = output[0]["generated_text"]
    return answer
