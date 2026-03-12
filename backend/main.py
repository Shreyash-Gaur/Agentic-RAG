# backend/main.py
"""
FastAPI entry point — updated alongside the agentic agent rewrite.

Fixes vs original:
  1. sources=[] hardcoded → now populated from agent's document_sources.
  2. mode inferred from max_tokens → now read directly from request field.
  3. CORS: removed allow_credentials=True when allow_origins=["*"]
     (that combination is rejected by browsers for credentialed requests).
  4. All bare `except:` replaced with `except Exception:`.
"""

from __future__ import annotations

import os
import sys
import time
import shlex
import logging
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# --- Core ---
from backend.core.config   import settings
from backend.core.logger   import setup_logging
from backend.core.exceptions import AgenticRAGException

# --- Services ---
from backend.services.retrieve_service      import RetrieveService
from backend.services.embed_cache_service   import EmbedCacheService
from backend.services.memory_service        import MemoryService
from backend.services.semantic_cache_service import SemanticCacheService

# --- Tools ---
from backend.tools.reranker import Reranker

# --- Agents ---
from backend.agents.graph_agent import GraphRAGAgent

# --- Models ---
from backend.models.request_models  import QueryRequest, RetrieveRequest
from backend.models.response_models import (
    QueryResponse, RetrieveResponse, DocumentResult
)

setup_logging()
logger = logging.getLogger("agentic-rag.api")

# ---------------------------------------------------------------------------
# Global service singletons
# ---------------------------------------------------------------------------

retrieve_service: Optional[RetrieveService]       = None
embed_cache:      Optional[EmbedCacheService]      = None
memory_service:   Optional[MemoryService]          = None
semantic_cache:   Optional[SemanticCacheService]   = None
rag_agent:        Optional[GraphRAGAgent]          = None


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory_service, embed_cache, retrieve_service, semantic_cache, rag_agent

    logger.info("Starting Agentic-RAG service...")

    # 1. Memory
    try:
        memory_service = MemoryService()
        logger.info("MemoryService ready.")
    except Exception as e:
        logger.exception("MemoryService init failed: %s", e)

    # 2. Embed cache
    try:
        embed_cache = EmbedCacheService()
        logger.info("EmbedCacheService ready.")
    except Exception as e:
        logger.exception("EmbedCacheService init failed: %s", e)

    # 3. Reranker
    reranker_obj = None
    if settings.RERANKER_ENABLED:
        try:
            reranker_obj = Reranker()
            logger.info("Reranker loaded: %s", settings.RERANKER_MODEL)
        except Exception as e:
            logger.exception("Reranker init failed: %s", e)

    # 4. Semantic cache
    try:
        semantic_cache = SemanticCacheService()
        logger.info("SemanticCacheService ready.")
    except Exception as e:
        logger.exception("SemanticCacheService init failed: %s", e)

    # 5. Retriever
    try:
        retrieve_service = RetrieveService(
            embed_cache=embed_cache,
            reranker_obj=reranker_obj,
            reranker_enabled=settings.RERANKER_ENABLED,
        )
        logger.info("RetrieveService ready (index=%s).", settings.FAISS_INDEX_PATH)
    except Exception as e:
        logger.exception("RetrieveService init failed: %s", e)

    # 6. Agent (compiled once here — NOT rebuilt on every request)
    if retrieve_service:
        try:
            rag_agent = GraphRAGAgent(
                retrieve_service=retrieve_service,
                model_name=settings.OLLAMA_MODEL,
            )
            logger.info("GraphRAGAgent ready.")
        except Exception as e:
            logger.exception("GraphRAGAgent init failed: %s", e)
    else:
        logger.warning("RetrieveService missing — GraphRAGAgent not started.")

    yield  # ── app running ──────────────────────────────────────────────────

    logger.info("Shutting down Agentic-RAG service...")
    for svc in [retrieve_service, embed_cache, memory_service]:
        if svc and hasattr(svc, "close"):
            try:
                svc.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

APP = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

# FIX: allow_origins=["*"] + allow_credentials=True is rejected by browsers.
# If you need credentials (cookies / auth headers), replace "*" with an
# explicit list of origins, e.g. ["http://localhost:3000"].
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # ← was True (invalid with wildcard origin)
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@APP.get("/health")
def health():
    return {
        "status":         "ok",
        "retriever":      bool(retrieve_service),
        "rag_agent":      bool(rag_agent),
        "semantic_cache": bool(semantic_cache),
        "memory_service": bool(memory_service),
    }


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

@APP.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest):
    if not retrieve_service:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    try:
        raw_results = retrieve_service.retrieve(req.query, top_k=req.top_k)
        doc_results = []
        for r in raw_results:
            meta = r.get("meta", {})
            doc_results.append(DocumentResult(
                text=meta.get("text", ""),
                score=r.get("score", 0.0),
                metadata=meta,
                source=meta.get("pdf") or meta.get("doc_name") or "unknown",
                chunk_id=r.get("index"),
            ))
        return RetrieveResponse(
            query=req.query,
            results=doc_results,
            num_results=len(doc_results),
        )
    except Exception as e:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Query  (main agentic endpoint)
# ---------------------------------------------------------------------------

@APP.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")

    try:
        session_id = req.conversation_id or "default"
        user_query = req.query

        # 1. Semantic cache check
        # Skip cache when ANY of these are true:
        #   - mode is not concise (long answer button sends mode=detailed)
        #   - temperature > 0.1 (creative answer button)
        #   - max_tokens above default (long answer sends 2x MAX_TOKENS)
        #   - bypass_cache field present and True
        use_cache = (
            semantic_cache
            and req.mode == "concise"
            and req.temperature <= 0.1
            and req.max_tokens <= settings.MAX_TOKENS
            and not getattr(req, "bypass_cache", False)
        )
        if use_cache:
            cached = semantic_cache.check_cache(user_query)
            if cached:
                logger.info("Serving from semantic cache.")
                if memory_service:
                    memory_service.add_turn(session_id, user_query, cached)
                return QueryResponse(
                    query=user_query,
                    answer=cached,
                    sources=[],
                    num_sources=0,
                    prompt="",
                    metadata={"source": "semantic_cache", "steps": ["cache_hit"]},
                )

        # 2. Memory context
        chat_history = ""
        if memory_service:
            chat_history = memory_service.get_context(session_id)

        # 3. Run agent
        # FIX: mode now comes directly from the request field, not inferred
        # from max_tokens.
        output = rag_agent.query(
            query=user_query,
            mode=req.mode,                   # ← explicit field
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            chat_history=chat_history,
        )

        ai_answer  = output.get("answer", "No answer generated.")
        raw_sources = output.get("sources", [])

        # 4. Persist to memory + semantic cache
        if memory_service:
            memory_service.add_turn(session_id, user_query, ai_answer)
        if semantic_cache and req.mode == "concise" and not getattr(req, "bypass_cache", False):
            semantic_cache.add_new_turn(user_query, ai_answer)

        # 5. Build source list for response
        # FIX: was always sources=[], num_sources=0 — now populated.
        doc_results = []
        for meta in raw_sources:
            if not meta:
                continue
            doc_results.append(DocumentResult(
                text=meta.get("text", "")[:500],  # truncate for response size
                score=meta.get("score", 0.0),
                metadata=meta,
                source=meta.get("pdf") or meta.get("doc_name") or "unknown",
                chunk_id=meta.get("chunk_id"),
            ))

        return QueryResponse(
            query=user_query,
            answer=ai_answer,
            sources=doc_results,
            num_sources=len(doc_results),
            prompt="",
            metadata=output.get("metadata", {}),
        )

    except Exception as e:
        logger.exception("Agent query failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Upload + ingestion
# ---------------------------------------------------------------------------

INGEST_UPLOAD_DIR = Path(settings.WATCH_DIR)
INGEST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _run_ingest_subprocess(
    cmd_list: List[str],
    env: Optional[Dict[str, str]] = None,
    log_prefix: str = "ingest",
) -> Dict:
    repo_root = Path(__file__).resolve().parents[1]
    log_dir   = repo_root / "logs" / "ingests"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts       = int(time.time())
    out_log  = log_dir / f"{log_prefix}-{ts}.out.log"
    err_log  = log_dir / f"{log_prefix}-{ts}.err.log"
    full_env = {**os.environ, **(env or {})}

    if "PYTHONPATH" not in full_env:
        full_env["PYTHONPATH"] = str(repo_root)

    with open(out_log, "ab") as out_f, open(err_log, "ab") as err_f:
        p = subprocess.Popen(
            cmd_list,
            env=full_env,
            stdout=out_f,
            stderr=err_f,
            start_new_session=True,
        )

    return {
        "pid":     p.pid,
        "out_log": str(out_log),
        "err_log": str(err_log),
        "cmd":     " ".join(shlex.quote(x) for x in cmd_list),
    }


@APP.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    filename = Path(file.filename).name
    out_path = INGEST_UPLOAD_DIR / filename

    with out_path.open("wb") as fh:
        fh.write(await file.read())

    repo_root   = Path(__file__).resolve().parents[1]
    script_path = repo_root / "backend" / "scripts" / "ingest_multi_docs.py"

    info = _run_ingest_subprocess(
        cmd_list=[
            sys.executable, str(script_path),
            "--input",       str(out_path.resolve()),
            "--out-index",   settings.FAISS_INDEX_PATH,
            "--out-meta",    settings.FAISS_META_PATH,
            "--chunk-tokens", str(settings.CHUNK_TOKENS),
            "--overlap",     str(settings.CHUNK_OVERLAP),
            "--batch",       str(settings.EMBEDDING_BATCH_SIZE),
            "--append",
        ],
        env={"PYTHONPATH": str(repo_root)},
        log_prefix="upload_ingest",
    )

    return {
        "status":   "accepted",
        "filename": filename,
        "pid":      info["pid"],
        "logs":     info["out_log"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:APP", host="0.0.0.0", port=8000, reload=settings.DEBUG)