"""
Embedding generation using Ollama.
"""

import requests
from typing import List
from core.config import settings
from core.logger import setup_logger
from core.exceptions import EmbeddingError

logger = setup_logger(__name__)


class Embedder:
    """
    Generate embeddings using Ollama embedding models.
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        """
        Initialize embedder.
        
        Args:
            base_url: Ollama API base URL
            model: Embedding model name
        """
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.EMBEDDING_MODEL
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            url = f"{self.base_url}/api/embeddings"
            
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("embedding", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding API error: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return embeddings

