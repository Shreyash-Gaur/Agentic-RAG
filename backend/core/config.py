# backend/core/config.py

"""
Central configuration using Pydantic Settings.
Loads variables from .env and provides safe defaults.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # App
    API_TITLE: str = "Agentic RAG API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
    ]

    # Ollama / LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = ""  # optional
    EMBEDDING_MODEL: str = "mxbai-embed-large:latest"

    # RAG Behavior
    TOP_K_RETRIEVAL: int = 5
    CHUNK_TOKENS: int = 512
    CHUNK_OVERLAP: int = 128

    # Vector store paths
    FAISS_INDEX_PATH: str = "backend/db/book_king_faiss.index"
    FAISS_META_PATH: str = "backend/db/book_king_meta.jsonl"

    # Memory
    MEMORY_DB_PATH: str = "backend/db/memory_store.sqlite"
    MEMORY_MAX_TURNS: int = 20

    # Reranker
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_INITIAL_K: int = 20

    # Embedding cache (new)
    EMBEDDING_CACHE_DB: str = "backend/db/embed_cache.sqlite"

    # Chainlit
    CHAINLIT_ENABLED: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

