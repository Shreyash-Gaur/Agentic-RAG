import logging
from typing import List, Dict, Any

from backend.core.config import settings

logger = logging.getLogger("agentic-rag.reranker")

class Reranker:
    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        self.model_name = model_name
        self.enabled = settings.RERANKER_ENABLED
        if self.enabled:
            try:
                from FlagEmbedding import FlagReranker
                logger.info(f"Loading reranker model: {model_name}...")
                self.reranker = FlagReranker(model_name, use_fp16=True)
            except ImportError:
                logger.warning("FlagEmbedding not installed. Reranking disabled.")
                self.enabled = False
                self.reranker = None
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                self.enabled = False
                self.reranker = None
                
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled or not self.reranker or not documents:
            return documents[:top_k]
            
        texts = [doc.get("meta", {}).get("text", "") for doc in documents]
        pairs = [[query, txt] for txt in texts]
        
        try:
            scores = self.reranker.compute_score(pairs)
            
            # Tie scores back to documents
            for i, doc in enumerate(documents):
                doc["score"] = float(scores[i])
                
            # Sort by score descending
            documents.sort(key=lambda x: x["score"], reverse=True)
            return documents[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking computation failed: {e}")
            return documents[:top_k]