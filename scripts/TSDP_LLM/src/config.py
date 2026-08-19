from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    BOOKS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "books")
    QDRANT_PATH: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "qdrant_db")
    COLLECTION_NAME: str = "football_knowledge"

    # Ingestion & Chunking
    CHUNK_SIZE: int = 500  # approximate words / tokens
    CHUNK_OVERLAP: int = 80

    # Embedding Model (FastEmbed)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

    # Local LLM (Ollama)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"

    # OpenRouter Free LLM
    OPENROUTER_API_KEYS: str = ""
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"

    @property
    def api_keys_list(self) -> List[str]:
        if not self.OPENROUTER_API_KEYS:
            return []
        return [k.strip() for k in self.OPENROUTER_API_KEYS.split(",") if k.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.BOOKS_DIR.mkdir(parents=True, exist_ok=True)
settings.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
