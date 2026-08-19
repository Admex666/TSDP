# Tasks & Roadmap

## Completed Tasks (Phase 1: Knowledge Ingestion)
- [x] **Task 1.1**: Setup project dependencies and configuration structure (`requirements.txt`, `.env.example`, `config.py`).
- [x] **Task 1.2**: Implement PDF extractor with page numbering and metadata preservation (`src/ingestion/pdf_parser.py`).
- [x] **Task 1.3**: Implement custom sentence/sliding-window chunker with page-boundary tracking (`src/ingestion/chunker.py`).
- [x] **Task 1.4**: Implement vector storage & embedding indexer with Qdrant and FastEmbed (`src/ingestion/vector_store.py`).
- [x] **Task 1.5**: Build batch ingestion pipeline script (`src/ingest.py`).
- [x] **Task 1.6**: Add end-to-end tests/verification script with sample football tactics PDF (`tests/test_ingestion.py`).

## Active Tasks (Phase 2: RAG Engine & Dual LLMs)
- [ ] **Task 2.1**: Implement local Ollama client (`src/llm/ollama_client.py`).
- [ ] **Task 2.2**: Implement OpenRouter free API client with multi-key rotation (`src/llm/openrouter_client.py`).
- [ ] **Task 2.3**: Implement unified LLM interface and RAG orchestration (`src/rag/engine.py`).
- [ ] **Task 2.4**: Benchmark & comparison mode between Ollama and OpenRouter.

## Upcoming Tasks (Phase 3: Streamlit UI)
- [ ] **Task 3.1**: Build Streamlit Chat Interface (`app.py`).
- [ ] **Task 3.2**: Source citation view (expandable book and page references).
- [ ] **Task 3.3**: Side-by-side backend comparator view.
