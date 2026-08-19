# Architectural & Technical Decisions

## 1. Embedding Engine: FastEmbed (ONNX) with BGE / MiniLM
- **Decision**: Use `fastembed` with `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`.
- **Rationale**: `fastembed` uses ONNX Runtime directly, requiring zero heavy PyTorch dependencies, taking minimal RAM (<200MB) and executing embeddings blazingly fast on both CPU (i5-10400F) and GPU.
- **Impact**: Fast PDF ingestion without memory spikes on the 16GB system.

## 2. Vector Database: Embedded Qdrant (Local Storage Mode)
- **Decision**: Use `qdrant-client` directly in local filesystem storage mode (`QdrantClient(path="./data/qdrant_db")`).
- **Rationale**: Avoids requiring Docker/Docker Desktop daemon, operates as a single lightweight Python embedded DB with full payload filtering and vector indexing capabilities.
- **Impact**: Zero-setup local persistence, high portability.

## 3. Local LLM Backend: Ollama with Lightweight 4-bit Quantized Model
- **Decision**: Recommend `qwen2.5:3b` (ultra fast, ~2GB VRAM) or `qwen2.5:7b-instruct-q4_K_M` (~4.5GB split / CPU offload) or `llama3.2:3b`.
- **Rationale**: User hardware has GTX 1650 Super (4GB VRAM). 3B/7B quantized models run smoothly without overloading GPU memory while providing high tactical reasoning quality.
- **Impact**: Smooth local execution without out-of-memory crashes.

## 4. Free Cloud LLM Backend: OpenRouter Free Models with Key Rotation
- **Decision**: Use OpenRouter free endpoint models (e.g. `meta-llama/llama-3.3-70b-instruct:free`, `google/gemini-2.0-flash-lite:free`, `qwen/qwen-2.5-72b-instruct:free` or `openrouter/free`) and implement round-robin / fallback key rotation.
- **Rationale**: Free tier endpoints have rate limits per minute/day; key rotation maximizes availability without cost.
- **Impact**: Reliable free cloud inference for benchmark comparison.

## 5. No LangChain / Framework Bloat
- **Decision**: Implement custom pure Python pipeline for chunking, retrieval, prompt assembly, and backend abstraction.
- **Rationale**: Eliminates hidden abstraction overhead, breaking changes, and debug friction.
- **Impact**: Clean, predictable, maintainable codebase.
