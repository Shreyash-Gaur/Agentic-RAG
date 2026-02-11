# backend/services/retrieve_service.py
"""
RetrieveService
- Wraps a FAISS index + metadata JSONL
- Uses EmbedCacheService to avoid re-embedding identical texts
- Falls back to repo Embedder if cache misses
- Optional reranker object can be provided (expects reranker.rerank_results(query, candidates, top_k))
"""

from __future__ import annotations
import json
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss

# absolute imports to be robust
from backend.services.embed_cache_service import EmbedCacheService
from backend.services.memory_service import MemoryService  # not used directly but useful later
from backend.core.config import settings
from backend.tools.embedder import Embedder
from backend.tools.retriever_faiss import FAISSRetriever  # fallback usage / sanity

logger = logging.getLogger("agentic-rag.retrieve")

# Default paths (from config if available)
DEFAULT_INDEX = settings.FAISS_INDEX_PATH if hasattr(settings, "FAISS_INDEX_PATH") else "backend/db/vector_data/data_faiss.index"
DEFAULT_META = settings.FAISS_META_PATH if hasattr(settings, "FAISS_META_PATH") else "backend/db/vector_data/data_meta.jsonl"
DEFAULT_EMBED_MODEL = settings.EMBEDDING_MODEL if hasattr(settings, "EMBEDDING_MODEL") else "mxbai-embed-large:latest"


def _load_meta_lines(meta_path: str) -> List[Dict[str, Any]]:
    meta = []
    if not os.path.exists(meta_path):
        return meta
    with open(meta_path, "r", encoding="utf8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                meta.append(json.loads(ln))
            except Exception:
                # tolerate non-json lines
                meta.append({"text": ln})
    return meta


class RetrieveService:
    def __init__(
        self,
        index_path: str = DEFAULT_INDEX,
        meta_path: str = DEFAULT_META,
        embed_cache: Optional[EmbedCacheService] = None,
        embedder: Optional[Embedder] = None,
        reranker_obj: Optional[Any] = None,
        reranker_enabled: bool = False,
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.reranker = reranker_obj
        self.reranker_enabled = bool(reranker_enabled)

        # load metadata
        self.meta: List[Dict[str, Any]] = _load_meta_lines(self.meta_path)
        logger.info("Loaded metadata entries: %d", len(self.meta))

        # load faiss index lazily
        self._index: Optional[faiss.Index] = None
        self._index_ntotal: int = 0

        # embedder / cache
        self.embedder = embedder or Embedder()
        self.embed_model = getattr(self.embedder, "model", DEFAULT_EMBED_MODEL)
        self.embed_cache = embed_cache or EmbedCacheService()

        # Check index exists
        if os.path.exists(self.index_path):
            try:
                self._index = faiss.read_index(self.index_path)
                self._index_ntotal = int(self._index.ntotal)
                logger.info("FAISS index loaded. ntotal=%d", self._index_ntotal)
            except Exception as e:
                logger.exception("Failed to load FAISS index: %s", e)
                self._index = None
        else:
            logger.warning("FAISS index not found at %s", self.index_path)
            self._index = None

    # --------------------------
    # Embedding helpers (with cache)
    # --------------------------
    def embed_text(self, text: str) -> np.ndarray:
        """
        Return a single embedding vector, using cache if available.
        """
        # check cache
        vec = self.embed_cache.get_vector(text, self.embed_model)
        if vec is not None:
            return vec.astype("float32")
        # else compute via embedder
        vec = self.embedder.embed_batch([text])[0]
        # persist
        try:
            self.embed_cache.set_vector(text, self.embed_model, vec)
        except Exception:
            logger.exception("Failed to write embed cache")
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batch embedding using cache where possible. Returns np.ndarray shape (n, dim)
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        # fast path: ask cache for all
        cached_vecs, keys = self.embed_cache.get_batch(texts, self.embed_model)
        miss_idx = [i for i, v in enumerate(cached_vecs) if v is None]
        results = [v if v is None else np.asarray(v, dtype=np.float32) for v in cached_vecs]

        if miss_idx:
            to_embed = [texts[i] for i in miss_idx]
            try:
                embedded = self.embedder.embed_batch(to_embed)
                embedded = [np.asarray(v, dtype=np.float32) for v in embedded]
            except Exception:
                # fallback: embed one by one
                embedded = [self.embedder.embed(t) for t in to_embed]
                embedded = [np.asarray(v, dtype=np.float32) for v in embedded]

            # write back to cache and fill results
            for idx_local, vec in enumerate(embedded):
                i_global = miss_idx[idx_local]
                results[i_global] = vec
            try:
                # write batch to cache
                self.embed_cache.set_batch(to_embed, self.embed_model, embedded)
            except Exception:
                logger.exception("Failed to set batch in embed cache")

        # stack into np.ndarray
        stacked = np.vstack([np.asarray(v, dtype=np.float32) for v in results])
        return stacked

    # --------------------------
    # Low-level index search
    # --------------------------
    def _ensure_index(self):
        if self._index is None:
            if os.path.exists(self.index_path):
                try:
                    self._index = faiss.read_index(self.index_path)
                    self._index_ntotal = int(self._index.ntotal)
                except Exception as e:
                    logger.exception("Failed to load FAISS index lazily: %s", e)
                    self._index = None

    def search_vector(self, vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query FAISS index with a single vector. Returns distances and indices arrays.
        """
        self._ensure_index()
        if self._index is None:
            raise RuntimeError("FAISS index not loaded")
        q = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        D, I = self._index.search(q, top_k)
        return D, I

    # --------------------------
    # High-level retrieve APIs
    # --------------------------
    def _build_result_record(self, idx: int, score: float) -> Dict[str, Any]:
        meta = self.meta[idx] if 0 <= idx < len(self.meta) else {}
        return {"index": idx, "score": float(score), "meta": meta}

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k chunks for the query. If reranker is configured and enabled,
        retrieve a larger candidate set, call reranker, and return top_k results.
        """
        # embed query
        qvec = self.embed_text(query).astype("float32")

        # if reranker enabled, fetch initial candidates larger than top_k
        if self.reranker_enabled and self.reranker:
            initial_k = max(getattr(settings, "RERANKER_INITIAL_K", 20), top_k)
            D, I = self.search_vector(qvec, initial_k)
            candidates = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                candidates.append(self._build_result_record(int(idx), float(dist)))
            try:
                reranked = self.reranker.rerank(query, candidates, top_k=top_k)
                return reranked
            except Exception:
                logger.exception("Reranker failed, falling back to original scores")
                # fallback to truncated candidates
                return candidates[:top_k]
        else:
            D, I = self.search_vector(qvec, top_k)
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                results.append(self._build_result_record(int(idx), float(dist)))
            return results

    def retrieve_batch(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieval: embed queries in batch (uses cache) and search each vector.
        Returns list-of-lists of results aligned with queries.
        """
        if not queries:
            return [[] for _ in queries]

        vecs = self.embed_batch(queries)
        out = []
        for v in vecs:
            D, I = self.search_vector(v, top_k)
            res = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                res.append(self._build_result_record(int(idx), float(dist)))
            out.append(res)
        return out

    # --------------------------
    # Utilities
    # --------------------------
    def reload_meta(self):
        self.meta = _load_meta_lines(self.meta_path)

    def index_count(self) -> int:
        return self._index_ntotal

    def close(self):
        try:
            self.embed_cache.close()
        except Exception:
            pass
        # faiss index doesn't need explicit close
