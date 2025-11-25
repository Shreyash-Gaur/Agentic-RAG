# backend/services/retrieve_service.py
"""
RetrieveService: thin service wrapper around FAISS retriever + optional cross-encoder reranker.

Responsibilities:
 - encapsulate FAISS retrieval and reranking logic
 - provide single-query and batch-query APIs
 - provide a small abstraction so tests and agents can call retrieval without depending on FastAPI
"""

from typing import List, Dict, Any, Optional
import os
import logging

# local imports
from tools.retriever_faiss import FAISSRetriever

log = logging.getLogger(__name__)


class RetrieveService:
    def __init__(
        self,
        index_path: str,
        meta_path: str,
        reranker_obj: Optional[Any] = None,
        reranker_enabled: bool = False,
        reranker_initial_k: int = 20,
    ):
        """
        index_path: path to faiss index file
        meta_path: path to metadata jsonl file
        reranker_obj: object with rerank / rerank_results / score methods (optional)
        reranker_enabled: whether to call reranker
        reranker_initial_k: how many initial candidates to fetch from FAISS when reranking
        """
        self.retriever = FAISSRetriever(index_path, meta_path)
        self.reranker = reranker_obj
        self.reranker_enabled = reranker_enabled and (reranker_obj is not None)
        self.reranker_initial_k = int(os.getenv("RERANKER_INITIAL_K", reranker_initial_k))

        log.info(
            "RetrieveService initialized. index.ntotal=%s reranker_enabled=%s",
            self.retriever.index.ntotal,
            self.reranker_enabled,
        )

    # ----- Helpers to call reranker flexibly -----
    def _call_reranker_flexibly(self, query: str, candidates: List[Dict[str, Any]], top_k: int):
        """
        Accept different reranker method shapes:
          - rerank_results(query, candidates, top_k)
          - rerank(query, candidates, top_k)
          - score(query, list_of_texts) -> scores (then sort)
        """
        if not self.reranker:
            return candidates[:top_k]

        # prefer dedicated rerank methods first
        if hasattr(self.reranker, "rerank_results"):
            return self.reranker.rerank_results(query, candidates, top_k=top_k)
        if hasattr(self.reranker, "rerank"):
            return self.reranker.rerank(query, candidates, top_k=top_k)

        # fallback: score-based reranker
        if hasattr(self.reranker, "score"):
            texts = [c["meta"].get("text", "") for c in candidates]
            scores = self.reranker.score(query, texts)
            for c, s in zip(candidates, scores):
                c["_rerank_score"] = float(s)
            candidates_sorted = sorted(candidates, key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
            return candidates_sorted[:top_k]

        # final fallback
        return candidates[:top_k]

    # ----- Public APIs -----
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Single-query retrieval. If reranker is enabled, this will:
         - fetch initial_k = max(reranker_initial_k, top_k) candidates
         - rerank them and return top_k
        """
        if self.reranker_enabled:
            initial_k = max(self.reranker_initial_k, top_k)
            candidates = self.retriever.retrieve(query, top_k=initial_k)
            results = self._call_reranker_flexibly(query, candidates, top_k=top_k)
        else:
            results = self.retriever.retrieve(query, top_k=top_k)
        return results

    def retrieve_batch(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieval for many queries. Behavior:
         - If reranker is disabled: run fast batch search and return top_k per query.
         - If reranker is enabled: we still use batch search to fetch initial_k candidates for each query,
           then rerank per-query. (We do per-query reranking to keep semantics clear.)
        """
        if not queries:
            return []

        if self.reranker_enabled:
            initial_k = max(self.reranker_initial_k, top_k)
            initial_batch = self.retriever.retrieve_batch(queries, top_k=initial_k)
            # per-query rerank
            final_batch = []
            for q, cand_list in zip(queries, initial_batch):
                reranked = self._call_reranker_flexibly(q, cand_list, top_k=top_k)
                final_batch.append(reranked)
            return final_batch
        else:
            return self.retriever.retrieve_batch(queries, top_k=top_k)

    def info(self) -> Dict[str, Any]:
        """Basic health / diagnostic info"""
        return {
            "ntotal": getattr(self.retriever.index, "ntotal", None),
            "reranker_enabled": self.reranker_enabled,
            "reranker": type(self.reranker).__name__ if self.reranker else None,
        }
