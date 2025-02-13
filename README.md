# ClauseRAG (Local RAG)

This repository provides an **Instant Legal Clause Finder** that uses a **Retrieval-Augmented Generation (RAG)** approach with **open-source models**. You can **upload** legal documents (PDF/DOCX), **retrieve** relevant clauses using a local vector store (FAISS), and generate answers with a local large language model (e.g., Falcon, Llama 2, Flan-T5).

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Requirements](#requirements)  
4. [Folder Structure](#folder-structure)  
5. [Installation & Setup](#installation--setup)  
6. [Usage](#usage)  
7. [How It Works](#how-it-works)  
8. [Customization](#customization)  
9. [Troubleshooting](#troubleshooting)  
10. [License & Disclaimer](#license--disclaimer)

---

## Overview

This system demonstrates how you can build an **end-to-end** RAG pipeline **locally** without relying on external APIs (like OpenAI). The pipeline is as follows:

1. **Ingestion**: Parse and chunk legal documents (PDF, DOCX) into smaller segments.  
2. **Embeddings**: Use SentenceTransformers (`all-MiniLM-L6-v2`) to embed each chunk.  
3. **Vector Store**: Store embeddings in a **FAISS** index.  
4. **Retrieval**: For each user query, embed the query and perform similarity search to find the most relevant chunks.  
5. **Generation**: Use a **local LLM** (e.g., Falcon, Llama 2, or Flan-T5) to craft a final answer based on the retrieved context.  
6. **Deployment**: Provide both a **CLI** (`main.py`) and a **Streamlit** UI (`streamlit_app.py`).

---

## Features

- **Upload any PDF or DOCX**: The app parses it on the fly.  
- **Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` by default.  
- **Local LLM**: Example code for “Falcon 7B Instruct” or “Flan-T5” (can be swapped for Llama 2, GPT-NeoX, MPT, etc.).  
- **Retrieval-Augmented**: High-quality answers that reference the document text.  
- **Streamlit UI**: Simple web interface to upload files, ask questions, and see immediate answers.  
- **CLI**: A straightforward console-based interface for ingestion/indexing/querying.

---

## Requirements

- Python 3.8+  
- A GPU (recommended) with sufficient VRAM if using a large model like Falcon or Llama. CPU-only usage is possible with smaller models, but can be slow.  

**Key Python libraries**:  
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF parsing  
- [python-docx](https://github.com/python-openxml/python-docx) for DOCX parsing  
- [SentenceTransformers](https://www.sbert.net/) for embeddings  
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search  
- [Transformers](https://github.com/huggingface/transformers) for local LLMs  
- [Streamlit](https://streamlit.io/) for the web interface (optional)

---

## Folder Structure

```bash
instant-legal-clause-finder/
├─ data/
│   ├─ contract.pdf
│   └─ contract.docx
├─ src/
│   ├─ ingest.py
│   ├─ create_index.py
│   ├─ query.py
│   └─ main.py
├─ streamlit_app.py
├─ all_chunks.pkl       # auto-created by ingest.py
├─ contracts.index      # auto-created by create_index.py
├─ metadata.pkl         # auto-created by create_index.py
├─ requirements.txt
└─ README.md
```
- **`data/`**: Place your legal PDFs/DOCX files here.  
- **`src/`**: Core Python scripts for ingestion and retrieval.  
  - `ingest.py`: Extract and chunk docs -> creates `all_chunks.pkl`  
  - `create_index.py`: Embeds chunks, builds FAISS index -> `contracts.index`, `metadata.pkl`  
  - `query.py`: Retrieval + generation logic  
  - `main.py`: CLI for queries  
- **`streamlit_app.py`**: A web UI for file upload + queries.  
- **`all_chunks.pkl`**, **`contracts.index`**, **`metadata.pkl`**: Generated artifacts.

---

## Installation & Setup

1. **Clone or download** this repository:  
   ```bash
   git clone https://github.com/yourusername/instant-legal-clause-finder.git
   cd instant-legal-clause-finder

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   .venv\Scripts\activate.bat # Windows

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

## Usage

### 1. CLI Pipeline:

#### Ingest & Chunk:
```bash
python src/ingest.py
```
This extracts text from data/ PDF/DOCX files and creates **(all_chunks.pkl)**.

#### Create Embedding Index:

```bash
python src/create_index.py
```
This embeds all chunks, builds a FAISS index **(contracts.index)**, and saves metadata **(metadata.pkl)**.

#### Query via CLI:

```bash
python src/main.py
```
Enter your questions about the contracts (e.g., “What is the termination clause?”).
Press Ctrl+C or type "exit" to quit.

### 2.Streamlit Web UI:

If you want to upload docs on-the-fly (and build an ephemeral index each time):

```bash
streamlit run streamlit_app.py
```

 - You can upload a PDF/DOCX, wait for processing, then ask questions.
 - The answers are generated using your chosen local LLM.

## How It Works

### Extraction
- `ingest.py` reads each **PDF or DOCX**, strips whitespace, splits text into ~500-word chunks with 50-word overlap, then pickles them.

### Embedding
- `create_index.py` uses **sentence-transformers/all-MiniLM-L6-v2** to embed each chunk (gives a vector).
- We store these vectors in **FAISS (`contracts.index`)** for fast nearest-neighbor lookups.

### Retrieval
- A user query is embedded similarly.
- We run a **top-K similarity search** in FAISS.
- The most relevant chunks are extracted from metadata.

### Generation
- We feed these **chunks + the user question** to a local **LLM (Falcon, Llama 2, or T5)**.
- The model returns a final **natural-language answer** that references the retrieved chunks.

### Answer
- Shown in **console (CLI)** or in the **Streamlit UI**.

---

## Customization

### Local LLM
- You can replace **`google/flan-t5-base`** with:
  - `tiiuae/falcon-7b-instruct`
  - `meta-llama/Llama-2-7b-chat-hf`
  - Any other Hugging Face model.
- Adjust the **pipeline parameters** accordingly.

### Embedding Model
- Default: `all-MiniLM-L6-v2`
- You can swap it for a **domain-specific model** (e.g., legal-based embeddings like `nlpaueb/legal-bert-base-uncased`) for better results on legal texts.

### Chunk Size & Overlap
- In `ingest.py`, tweak **`chunk_size`** and **`overlap`** if your documents need **larger/smaller chunks**.

### Vector DB
- If you have **many documents**, consider a more **scalable vector database**:
  - **Milvus**
  - **Chroma**
  - **Pinecone**
- The logic remains similar.

### Disclaimers
> **For real legal usage, ensure your disclaimers are visible:**
>  
> _“This system is not a substitute for professional legal advice.”_

---

## Troubleshooting

### Slow or Failing on CPU
- Large LLMs (**Falcon, Llama 2**) can be **slow on CPU**.
- Consider a **GPU** or a **smaller model** (e.g., **Flan-T5-Base**).

### `ModuleNotFoundError`
- Ensure you run commands **from the project root** and your environment is active.

### Out of Memory Errors:
- If you’re on GPU with limited VRAM, reduce the model size (7B → 3B or smaller).
- Try half-precision or 8-bit quantization if available.






