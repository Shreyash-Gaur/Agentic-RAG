"""
FAISS-based vector store retriever.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any
from core.config import settings
from core.logger import setup_logger
from core.exceptions import RetrievalError

logger = setup_logger(__name__)


class FAISSRetriever:
    """
    FAISS-based document retriever.
    """
    
    def __init__(self, index_path: str = None, embedder=None):
        """
        Initialize FAISS retriever.
        
        Args:
            index_path: Path to FAISS index directory
            embedder: Embedder instance for query embeddings
        """
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.embedder = embedder
        self.index = None
        self.metadata = []
        self.dimension = None
    
    def load_index(self):
        """Load FAISS index from disk."""
        try:
            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.pkl"
            
            if not index_file.exists():
                logger.warning(f"Index not found at {index_file}, creating new index")
                return
            
            self.index = faiss.read_index(str(index_file))
            self.dimension = self.index.dimension
            
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
            
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise RetrievalError(f"Failed to load index: {e}")
    
    def save_index(self):
        """Save FAISS index to disk."""
        try:
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.pkl"
            
            faiss.write_index(self.index, str(index_file))
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"Saved FAISS index to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise RetrievalError(f"Failed to save index: {e}")
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict] = None):
        """
        Add documents to the index.
        
        Args:
            texts: List of document texts
            embeddings: List of embedding vectors
            metadata: Optional metadata for each document
        """
        try:
            if not embeddings:
                raise ValueError("No embeddings provided")
            
            dimension = len(embeddings[0])
            
            # Initialize index if needed
            if self.index is None:
                self.dimension = dimension
                self.index = faiss.IndexFlatL2(dimension)
            
            # Convert to numpy array
            vectors = np.array(embeddings, dtype=np.float32)
            self.index.add(vectors)
            
            # Store metadata
            if metadata is None:
                metadata = [{}] * len(texts)
            
            for i, (text, meta) in enumerate(zip(texts, metadata)):
                self.metadata.append({
                    "text": text,
                    "metadata": meta,
                    "id": len(self.metadata)
                })
            
            logger.info(f"Added {len(texts)} documents to index")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise RetrievalError(f"Failed to add documents: {e}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("Index is empty, loading...")
                self.load_index()
            
            if self.index is None or self.index.ntotal == 0:
                return []
            
            if self.embedder is None:
                raise ValueError("Embedder not set")
            
            # Generate query embedding
            query_embedding = self.embedder.embed(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)
            
            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    doc = self.metadata[idx].copy()
                    doc["score"] = float(distance)
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise RetrievalError(f"Retrieval failed: {e}")


