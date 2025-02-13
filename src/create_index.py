# src/create_index.py

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    # 1. Load chunk dictionaries
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chunks_file = os.path.join(os.path.dirname(base_dir), "all_chunks.pkl")
    if not os.path.exists(chunks_file):
        print(f"File '{chunks_file}' not found. Run ingest.py first.")
        return

    with open(chunks_file, "rb") as f:
        all_chunks = pickle.load(f)  # list of dicts, each has "filename", "chunk_id", "text", etc.

    print(f"Loaded {len(all_chunks)} chunks from {chunks_file}")

    # 2. Initialize a local embedding model
    print("Loading SentenceTransformer model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 3. Convert each chunk's text to an embedding
    chunk_texts = [d["text"] for d in all_chunks]
    print("Generating embeddings for chunks...")
    chunk_embeddings = embedder.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)

    # 4. Build the FAISS index
    embeddings_matrix = chunk_embeddings.astype('float32')  # ensure float32
    dim = embeddings_matrix.shape[1]
    print(f"Building FAISS index with dimension: {dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_matrix)

    # 5. Save the index and metadata
    index_file = os.path.join(os.path.dirname(base_dir), "contracts.index")
    meta_file = os.path.join(os.path.dirname(base_dir), "metadata.pkl")

    faiss.write_index(index, index_file)
    print(f"FAISS index saved to: {index_file}")

    with open(meta_file, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Metadata (chunk dicts) saved to: {meta_file}")

if __name__ == "__main__":
    main()
