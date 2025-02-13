# Instant Legal Clause Finder (Local RAG)

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

