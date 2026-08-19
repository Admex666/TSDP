# Project: Football Knowledge Base AI (TSDP_LLM)

## High-Level Overview
A **Football Knowledge Base AI** is a specialized Retrieval-Augmented Generation (RAG) system that builds a searchable local knowledge base from football literature (PDF books) and answers natural language football coaching/tactical/analytical questions with precise source citations (book title and page numbers).

## Core Goals
- Parse and chunk football literature PDFs preserving page number metadata.
- Generate high-quality local embeddings with minimal hardware overhead.
- Store embeddings and metadata in a local Qdrant vector database.
- Provide natural language RAG-based query processing.
- Support dual LLM backends on identical context for benchmark & comparison:
  1. **Local LLM**: via Ollama (e.g. `qwen2.5:3b` / `qwen2.5:7b` quantized for 4GB VRAM).
  2. **Free Cloud LLM**: via OpenRouter free tier models with API key rotation.
- Offer an intuitive Streamlit Chat UI with side-by-side comparison and clickable/clear source citations.

## Scope & Phases
- **V1 - Knowledge Ingestion Pipeline**: PDF extraction, chunking, local embeddings, Qdrant storage.
- **V2 - RAG Engine & Dual Backends**: Vector retrieval, prompt engineering, Ollama & OpenRouter clients with key rotation.
- **V3 - Chat User Interface**: Streamlit UI with conversation history, model switching, source citations.

## Key Technologies
- **Language & Runtime**: Python 3.13
- **PDF Extraction**: PyMuPDF (`fitz`)
- **Chunking**: Custom Python chunker (semantic/token-aware sliding window with metadata)
- **Embeddings**: `fastembed` (ONNX runtime, ultra-lightweight, CPU/GPU fast) with `BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2`
- **Vector Database**: Qdrant (local persistent storage mode `QdrantClient(path=...)`)
- **LLM Backends**: Ollama (local) & OpenRouter API (free tier with key rotation)
- **UI**: Streamlit
