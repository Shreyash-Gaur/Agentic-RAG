# backend/agents/researcher_agent.py
"""
ResearcherAgent
- Responsible for retrieving relevant passages for a given query.
- Uses RetrieveService (which wraps FAISS + reranker).
- Exposes:
    - research(query, top_k) -> List[dict]  (each dict has index, score, meta)
    - refine_query(query, context) -> str    (optional, uses simple heuristics)
"""

from typing import List, Dict, Any, Optional
import logging

log = logging.getLogger("agentic-rag.researcher")

class ResearcherAgent:
    def __init__(self, retriever_service, max_chunk_chars: int = 2000):
        """
        retriever_service: instance of services.RetrieveService
        """
        self.retriever = retriever_service
        self.max_chunk_chars = int(max_chunk_chars)

    def research(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Run retrieval (with reranking if retriever_service configured).
        Returns a list of result dicts with fields { index, score, meta }.
        """
        if not self.retriever:
            raise RuntimeError("RetrieverService not available in ResearcherAgent.")
        results = self.retriever.retrieve(query, top_k=top_k)
        return results

    def refine_query(self, query: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Simple refinement: if context exists, append a clarifying phrase.
        For advanced usage replace with an LLM-based rewrite step.
        """
        if not context:
            return query
        # small heuristic: keep it short
        return query.strip() + " (use the provided context to answer precisely)"
