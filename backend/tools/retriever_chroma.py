"""
ChromaDB-based vector store retriever.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any
from core.config import settings
from core.logger import setup_logger
from core.exceptions import RetrievalError

logger = setup_logger(__name__)


class ChromaRetriever:
    """
    ChromaDB-based document retriever.
    """
    
    def __init__(self, persist_dir: str = None, collection_name: str = "documents", embedder=None):
        """
        Initialize ChromaDB retriever.
        
        Args:
            persist_dir: Directory to persist ChromaDB data
            collection_name: Name of the collection
            embedder: Embedder instance for query embeddings
        """
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self.embedder = embedder
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB retriever with collection: {collection_name}")
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]] = None,
        metadata: List[Dict] = None,
        ids: List[str] = None
    ):
        """
        Add documents to the collection.
        
        Args:
            texts: List of document texts
            embeddings: Optional embeddings (if None, will use embedder)
            metadata: Optional metadata for each document
            ids: Optional document IDs
        """
        try:
            if embeddings is None:
                if self.embedder is None:
                    raise ValueError("Embedder required if embeddings not provided")
                embeddings = self.embedder.embed_batch(texts)
            
            if metadata is None:
                metadata = [{}] * len(texts)
            
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(texts))]
            
            # Convert embeddings to list of lists
            embeddings_list = [list(emb) for emb in embeddings]
            
            self.collection.add(
                embeddings=embeddings_list,
                documents=texts,
                metadatas=metadata,
                ids=ids
            )
            
            logger.info(f"Added {len(texts)} documents to ChromaDB")
            
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
            if self.embedder is None:
                raise ValueError("Embedder not set")
            
            # Generate query embedding
            query_embedding = self.embedder.embed(query)
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "text": results['documents'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "score": 1.0 - results['distances'][0][i] if results['distances'] else 0.0,
                        "id": results['ids'][0][i] if results['ids'] else None
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise RetrievalError(f"Retrieval failed: {e}")

