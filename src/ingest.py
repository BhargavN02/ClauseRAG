# src/ingest.py

import os
import fitz              # for PDF reading
import pickle
import logging
import signal
import sys
from docx import Document
from typing import List, Dict
from tqdm import tqdm
from contextlib import contextmanager

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Signal handling for graceful shutdown
def signal_handler(signum, frame):
    logger.warning("Received interrupt signal, cleaning up...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@contextmanager
def safe_tqdm(*args, **kwargs):
    progress = tqdm(*args, **kwargs)
    try:
        yield progress
    finally:
        progress.close()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + " "
    except Exception as e:
        logger.error(f"Failed to read PDF {pdf_path}: {e}")
    return text

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logger.error(f"Failed to read DOCX {docx_path}: {e}")
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into a list of overlapping chunks, each up to `chunk_size` words.
    """
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_file(file_path: str) -> List[Dict]:
    """
    Extract text from a single PDF/DOCX, chunk it, and return
    a list of dictionaries, each representing one chunk.
    """
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()

    text = ""
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        # NOTE: python-docx doesn't reliably parse old .doc files (only .docx).
        text = extract_text_from_docx(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_name}")
        return []

    # Clean up whitespace
    text = " ".join(text.split())

    # Split text into chunks
    chunks = chunk_text(text)

    # Build a list of chunk dictionaries
    chunk_dicts = []
    for i, c in enumerate(chunks):
        chunk_metadata = {
            "filename": file_name,
            "file_type": ext,
            "chunk_id": i,
            "text": c
        }
        chunk_dicts.append(chunk_metadata)

    return chunk_dicts

def save_chunks(chunks: List[Dict], output_path: str) -> None:
    """Safely saves chunk dictionaries to the given output file."""
    try:
        with open(output_path, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"Saved {len(chunks)} chunks to '{output_path}'")
    except Exception as e:
        logger.error(f"Could not save file {output_path}: {e}")

def main():
    """
    1. Finds PDF/DOC/DOCX in `data/`.
    2. For each file, extracts text, chunks it, and stores chunk dictionaries.
    3. Saves all chunk dictionaries to `all_chunks.pkl`.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    output_file = os.path.join(os.path.dirname(base_dir), "all_chunks.pkl")

    logger.info(f"Reading files from '{data_dir}'...")

    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return

    supported = {".pdf", ".doc", ".docx"}
    file_paths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not file_paths:
        logger.warning("No .pdf, .doc, or .docx files found in data/")
        return

    all_chunks = []

    with safe_tqdm(file_paths, desc="Processing Files") as pbar:
        for path in pbar:
            chunk_dicts = process_file(path)
            all_chunks.extend(chunk_dicts)
            # Optional: periodically save progress to avoid too much memory usage
            if len(all_chunks) % 1000 == 0:
                save_chunks(all_chunks, output_file)

    # Final save of all chunks
    save_chunks(all_chunks, output_file)

if __name__ == "__main__":
    main()
