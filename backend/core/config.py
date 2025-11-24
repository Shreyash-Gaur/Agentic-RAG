"""
Configuration management using Pydantic settings.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "Agentic RAG API"
    API_VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    
    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "faiss"  # or "chroma"
    FAISS_INDEX_PATH: str = "db/faiss_index"
    CHROMA_PERSIST_DIR: str = "db/chroma_db"
    
    # Database
    METADATA_DB_PATH: str = "db/metadata_store.db"
    
    # RAG Settings
    TOP_K_RETRIEVAL: int = 5
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Agent Settings
    MAX_ITERATIONS: int = 10
    TEMPERATURE: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


