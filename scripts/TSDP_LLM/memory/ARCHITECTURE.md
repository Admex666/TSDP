# System Architecture

```
TSDP_LLM/
├── data/
│   ├── books/              # Source PDF files
│   └── qdrant_db/          # Persistent local Qdrant vector database
├── memory/                 # Project memory
├── src/
│   ├── config.py           # Configuration management (.env, paths, model settings)
│   ├── ingestion/
│   │   ├── pdf_parser.py   # PyMuPDF extractor with page mapping
│   │   ├── chunker.py      # Semantic / sliding-window text chunker
│   │   └── vector_store.py # FastEmbed + Qdrant embedded client
│   ├── llm/
│   │   ├── base.py         # Abstract Base LLM Client
│   │   ├── ollama_client.py# Local Ollama client
│   │   └── openrouter.py   # OpenRouter client with key rotation
│   ├── rag/
│   │   ├── prompt.py       # Prompt templates & context assembly
│   │   └── engine.py       # RAG pipeline coordinator
│   └── ingest.py           # CLI ingestion script
├── app.py                  # Streamlit Chat UI
├── requirements.txt
└── .env.example
```

## Data Ingestion Flow (Phase 1)
```
[PDF Book in data/books/]
           │
           ▼
[PyMuPDF Parser] ────► Extracts text + page metadata
           │
           ▼
[Custom Chunker] ────► Splits into coherent chunks (with page numbers, doc ID)
           │
           ▼
[FastEmbed ONNX] ────► Generates dense vector embeddings (bge-small-en-v1.5)
           │
           ▼
[Qdrant Local DB] ───► Stores vectors + payload {text, book_title, page_start, page_end}
```

## RAG Query Flow (Phase 2 & 3)
```
[User Query]
     │
     ▼
[Embedding] ────► [Qdrant Similarity Search] ────► [Top-K Chunks + Metadata]
                                                           │
                                                           ▼
                                            [Prompt Context Assembler]
                                                           │
                                     ┌─────────────────────┴─────────────────────┐
                                     ▼                                           ▼
                             [Ollama Backend]                         [OpenRouter Backend]
                             (Local GPU/CPU)                           (Free Cloud Tier)
                                     │                                           │
                                     └─────────────────────┬─────────────────────┘
                                                           ▼
                                            [Formatted Response + Sources]
                                                           │
                                                           ▼
                                                    [Streamlit UI]
```
