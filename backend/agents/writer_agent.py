# backend/agents/writer_agent.py
"""
WriterAgent
- Responsible for turning retrieved context + user question into a final answer.
- Uses OllamaClient (HTTP client wrapper) or other generator.
- Exposes generate_answer(query, context, max_tokens, temperature)
"""

from typing import List, Dict, Any, Optional
import logging
import os
from backend.tools.ollama_client import OllamaClient

log = logging.getLogger("agentic-rag.writer")

class WriterAgent:
    def __init__(self, ollama_client: Optional[OllamaClient] = None, model: Optional[str] = None):
        # Ollama client wrapper (simple). If None, create one from environment.
        self.ollama = ollama_client or OllamaClient()
        self.model = model or os.getenv("OLLAMA_MODEL", None)

    def _build_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        instructions = (
            "You are an assistant. Use the context passages to answer the question. "
            "Give a concise answer and list the sources (chunk ids) you used."
        )
        ctx_parts = []
        for c in context:
            meta = c.get("meta", {})
            text = meta.get("text", "")[:2000]
            idx = c.get("index")
            ctx_parts.append(f"[chunk {idx}]\n{text}\n")
        context_block = "\n".join(ctx_parts)
        prompt = f"{instructions}\n\nCONTEXT:\n{context_block}\n\nQUESTION: {query}\n\nAnswer:"
        return prompt

    def generate_answer(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(query, context)
        # prefer Ollama generate via HTTP wrapper
        try:
            if self.model:
                # OllamaClient.generate expects model param
                text = self.ollama.generate(self.model, prompt, max_tokens=max_tokens, temperature=temperature)
            else:
                # If no model configured, try env inside ollama client
                text = self.ollama.generate(os.getenv("OLLAMA_MODEL", ""), prompt, max_tokens=max_tokens, temperature=temperature)
            return {"answer": text, "prompt": prompt, "success": True}
        except Exception as e:
            log.exception("WriterAgent generation failed: %s", e)
            # fallback: return prompt so caller can run alternate generation
            return {"answer": None, "prompt": prompt, "success": False, "error": str(e)}