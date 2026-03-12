# backend/models/request_models.py
"""
Request models — updated to add explicit `mode` field to QueryRequest.

FIX vs original: mode was previously inferred from max_tokens > 512, meaning
callers had no way to explicitly request detailed vs concise output. It is
now a first-class field with a sensible default.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from backend.core.config import settings


# ---------------------------------------------------------------------------
# Basic RAG query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request schema for /query endpoint."""
    query:           str
    conversation_id: Optional[str]  = "default"
    mode:            Literal["concise", "detailed"] = "concise"   # ← NEW explicit field
    max_tokens:      int   = Field(default=settings.MAX_TOKENS, ge=1, le=4096)
    temperature:     float = Field(default=0.0, ge=0.0, le=1.0)
    top_k:           int   = Field(default=settings.TOP_K_RETRIEVAL, ge=1, le=50)
    system_prompt:   Optional[str] = None
    bypass_cache:    bool  = False   # skip semantic cache — used by action buttons


# ---------------------------------------------------------------------------
# Retrieval-only request
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    """Request schema for /retrieve endpoint."""
    query: str
    top_k: int = Field(default=settings.TOP_K_RETRIEVAL, ge=1, le=50)


# ---------------------------------------------------------------------------
# Iterative agentic query (multi-step, kept for backwards compat)
# ---------------------------------------------------------------------------

class IterativeQueryRequest(BaseModel):
    """Request schema for agent-based iterative reasoning."""
    query:           str
    conversation_id: Optional[str] = "default"
    top_k:           int  = Field(default=settings.TOP_K_RETRIEVAL, ge=1, le=50)
    max_iterations:  int  = Field(default=settings.MAX_ITERATIONS, ge=1, le=10)
    temperature:     float = 0.0


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Request schema for ingesting a single PDF/doc."""
    file_path:    str
    chunk_tokens: int = Field(default=settings.CHUNK_TOKENS, ge=1, le=1024)
    overlap:      int = Field(default=settings.CHUNK_OVERLAP, ge=1, le=1024)


class BatchIngestRequest(BaseModel):
    """Request schema for ingesting multiple documents."""
    file_paths:   List[str]
    chunk_tokens: int = Field(default=settings.CHUNK_TOKENS, ge=1, le=1024)
    overlap:      int = Field(default=settings.CHUNK_OVERLAP, ge=1, le=1024)


# ---------------------------------------------------------------------------
# Embedding cache management
# ---------------------------------------------------------------------------

class CacheLookupRequest(BaseModel):
    """For testing if a text is already cached."""
    text: str