# backend/agents/rag_agent.py
"""
RAGAgent: orchestrates ResearcherAgent + WriterAgent + optional Memory service.

Public methods:
 - query(query, top_k=5) -> dict: { answer, sources, prompt, context }
 - chat(conv_id, query, top_k=5) -> dict: same + side-effect: store in MemoryService
"""

from typing import Optional, List, Dict, Any
import logging
import time

log = logging.getLogger("agentic-rag.rag")

class RAGAgent:
    def __init__(self, researcher, writer, memory=None, max_iterations: int = 1):
        """
        researcher: ResearcherAgent instance
        writer: WriterAgent instance
        memory: MemoryService instance (optional)
        """
        self.researcher = researcher
        self.writer = writer
        self.memory = memory
        self.max_iterations = int(max_iterations)

    def query(self, query: str, top_k: int = 5, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Basic RAG flow:
         1) research -> get top_k passages
         2) writer -> generate answer using passages
        """
        t0 = time.time()
        context = self.researcher.research(query, top_k=top_k)
        t_retrieval = time.time() - t0

        t1 = time.time()
        gen = self.writer.generate_answer(query, context, max_tokens=max_tokens, temperature=temperature)
        t_gen = time.time() - t1

        answer = gen.get("answer")
        prompt = gen.get("prompt")

        result = {
            "query": query,
            "answer": answer,
            "prompt": prompt,
            "sources": [{"index": r["index"], "score": r["score"], "meta": r.get("meta", {})} for r in context],
            "timings": {"retrieval_s": t_retrieval, "generation_s": t_gen}
        }
        return result

    def chat(self, conv_id: str, query: str, top_k: int = 5, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Query + store conversation turns in memory (if memory available)
        """
        out = self.query(query, top_k=top_k, max_tokens=max_tokens, temperature=temperature)
        answer = out.get("answer", "")
        if self.memory:
            try:
                self.memory.add_turn(conv_id, "user", query, meta={"ts": time.time()})
                self.memory.add_turn(conv_id, "assistant", answer, meta={"sources": out["sources"]})
            except Exception as e:
                log.warning("Failed to persist memory for conv=%s: %s", conv_id, e)
        return out
