# Changelog

## 2026-08-19
- **Phase 1 Implementation Complete (Knowledge Ingestion Pipeline)**:
  - Created configuration module [`src/config.py`](file:///e:/Data/TSDP/scripts/TSDP_LLM/src/config.py) and [`.env.example`](file:///e:/Data/TSDP/scripts/TSDP_LLM/.env.example).
  - Built PyMuPDF-based text extractor with page-level metadata tracking [`src/ingestion/pdf_parser.py`](file:///e:/Data/TSDP/scripts/TSDP_LLM/src/ingestion/pdf_parser.py).
  - Built sliding-window text chunker with page-boundary tracking [`src/ingestion/chunker.py`](file:///e:/Data/TSDP/scripts/TSDP_LLM/src/ingestion/chunker.py).
  - Built embedded Qdrant vector database manager with FastEmbed ONNX embedding integration [`src/ingestion/vector_store.py`](file:///e:/Data/TSDP/scripts/TSDP_LLM/src/ingestion/vector_store.py).
  - Built CLI batch PDF ingestion pipeline [`src/ingest.py`](file:///e:/Data/TSDP/scripts/TSDP_LLM/src/ingest.py).
  - Added end-to-end verification test suite [`tests/test_ingestion.py`](file:///e:/Data/TSDP/scripts/TSDP_LLM/tests/test_ingestion.py), passing with high semantic retrieval scores.
- **Project Initialization**:
  - Reviewed and analyzed `handover_plan.md`.
  - Established canonical project memory in `/memory`.
