# backend/tools/query_expander.py
"""
HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

The correct implementation:
  - Generate a HYPOTHETICAL answer to the query.
  - Embed ONLY the hypothetical answer (not the query).
  - Use that embedding for FAISS retrieval.

Why this works: a hypothetical answer written in document-style language
sits much closer in vector space to your actual stored documents than a
short user question does. The retrieved docs are therefore more relevant.

FIX vs original: the old code concatenated query + hyde_context into one
string before embedding, which diluted the benefit. This version returns
ONLY the hypothetical text — the caller should embed this alone.
"""

import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from backend.core.config import settings

logger = logging.getLogger("agentic-rag.hyde")


def generate_hyde_document(query: str) -> str:
    """
    Generate a short hypothetical document that would answer `query`.

    Returns the hypothetical text only — NOT the original query.
    The caller should embed this text alone for retrieval.

    Falls back to the original query string on failure so retrieval
    always has something to work with.
    """
    logger.info("HyDE: generating hypothetical document for: '%s'", query)
    try:
        llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0.1)
        prompt = (
            "Write a short, factual paragraph that directly answers the following "
            "question or explains the topic. Do NOT include introductory filler — "
            "just the factual content a reference document would contain.\n\n"
            f"Question: {query}\n"
            "Factual paragraph:"
        )
        response      = llm.invoke([HumanMessage(content=prompt)])
        hypothetical  = response.content.strip()
        logger.info("HyDE: generated %d characters.", len(hypothetical))
        return hypothetical

    except Exception as e:
        logger.error("HyDE generation failed, falling back to raw query: %s", e)
        return query  # safe fallback — raw query still works, just less precise