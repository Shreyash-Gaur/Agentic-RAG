# backend/main.py
# add near other imports at top of file
from fastapi import BackgroundTasks, UploadFile, File
import uuid
import subprocess
import shlex
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# import settings and services
from backend.core.config import settings
from backend.services.memory_service import MemoryService
from backend.services.embed_cache_service import EmbedCacheService
from backend.services.retrieve_service import RetrieveService

# optional reranker and ollama client
from backend.tools.ollama_client import OllamaClient

# Logging
logger = logging.getLogger("agentic-rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

APP = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# CORS
cors_origins = settings.CORS
if cors_origins:
    from fastapi.middleware.cors import CORSMiddleware
    APP.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Config paths
DEFAULT_INDEX = settings.FAISS_INDEX_PATH
DEFAULT_META = settings.FAISS_META_PATH

# Globals (singletons)
memory_service: Optional[MemoryService] = None
embed_cache: Optional[EmbedCacheService] = None
retrieve_service: Optional[RetrieveService] = None
reranker_obj = None
ollama_client: Optional[OllamaClient] = None

# Request/response models
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class RetrieveResponse(BaseModel):
    query: str
    results: list

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    max_tokens: int = 512
    temperature: float = 0.0

class QueryResponse(BaseModel):
    query: str
    answer: str
    prompt: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None


@APP.on_event("startup")
def startup_event():
    global memory_service, embed_cache, retrieve_service, reranker_obj, ollama_client

    logger.info("Starting Agentic-RAG service...")

    # Memory
    try:
        memory_service = MemoryService(
            max_history=settings.MEMORY_MAX_TURNS,
            use_sqlite=True,
            db_path=settings.MEMORY_DB_PATH,
            preload=False,  # don't preload everything by default
        )
        logger.info("MemoryService initialized: %s", settings.MEMORY_DB_PATH)
    except Exception as e:
        logger.exception("Failed to init MemoryService: %s", e)
        memory_service = None

    # Embed cache
    try:
        embed_cache = EmbedCacheService(db_path=settings.EMBEDDING_CACHE_DB)
        logger.info("EmbedCacheService initialized: %s", settings.EMBEDDING_CACHE_DB)
    except Exception as e:
        logger.exception("Failed to init EmbedCacheService: %s", e)
        embed_cache = None

    # Reranker (optional)
    if settings.RERANKER_ENABLED:
        try:
            from backend.tools.reranker import CrossEncoderReranker
            reranker_obj = CrossEncoderReranker(model_name=settings.RERANKER_MODEL)
            logger.info("Reranker loaded: %s", settings.RERANKER_MODEL)
        except Exception as e:
            logger.exception("Failed to load reranker: %s", e)
            reranker_obj = None

    # Ollama client (optional)
    try:
        if settings.OLLAMA_MODEL:
            ollama_client = OllamaClient(base_url=settings.OLLAMA_BASE_URL)
            logger.info("Ollama client ready at %s (model=%s)", settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL)
        else:
            ollama_client = None
    except Exception as e:
        logger.exception("Failed to init OllamaClient: %s", e)
        ollama_client = None

    # RetrieveService: pass embed_cache and reranker if available
    try:
        retrieve_service = RetrieveService(
            index_path=DEFAULT_INDEX,
            meta_path=DEFAULT_META,
            embed_cache=embed_cache,
            embedder=None,  # Use default Embedder inside RetrieveService
            reranker_obj=reranker_obj,
            reranker_enabled=bool(reranker_obj is not None),
        )
        logger.info("RetrieveService initialized (index=%s)", DEFAULT_INDEX)
    except Exception as e:
        logger.exception("Failed to init RetrieveService: %s", e)
        retrieve_service = None


@APP.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down Agentic-RAG service...")
    try:
        if retrieve_service:
            retrieve_service.close()
    except Exception:
        pass
    try:
        if embed_cache:
            embed_cache.close()
    except Exception:
        pass
    try:
        if memory_service:
            memory_service.close()
    except Exception:
        pass


@APP.get("/health")
def health():
    return {
        "status": "ok",
        "retriever_loaded": retrieve_service is not None,
        "reranker_loaded": reranker_obj is not None,
        "ollama_configured": bool(settings.OLLAMA_MODEL),
    }

# ------------------------------
# Ingest endpoints (file upload / directory trigger)
# ------------------------------

INGEST_UPLOAD_DIR = Path("data/book")
INGEST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _run_ingest_subprocess(cmd: str, env: Optional[Dict[str, str]] = None) -> None:
    """
    Run ingestion command in a detached subprocess.
    We use subprocess.Popen to avoid blocking; logs will be printed to server stdout/stderr.
    """
    # Ensure we run from repo root and have PYTHONPATH
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    # Run in background (no wait)
    # Use shell=False with simple list to avoid shell-injection risks
    subprocess.Popen(shlex.split(cmd), env=full_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

@APP.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a single file and start ingestion in background.
    Saves to data/book/<filename> and starts ingest script for that file.
    """
    # save file to data/book
    filename = Path(file.filename).name
    out_path = INGEST_UPLOAD_DIR / filename
    # overwrite existing
    with out_path.open("wb") as fh:
        content = await file.read()
        fh.write(content)
    # build command to ingest single file; use the single-file script in backend/scripts
    # Use PYTHONPATH=. so backend package imports work
    cmd = f"PYTHONPATH=. python backend/scripts/ingest_book_king_faiss.py --pdf \"{out_path.resolve()}\" --chunk-tokens {settings.CHUNK_TOKENS} --overlap {settings.CHUNK_OVERLAP} --batch 64"
    # run in background
    _run_ingest_subprocess(cmd)
    return {"status": "accepted", "filename": filename, "ingest_cmd": cmd}

@APP.post("/ingest/dir")
def ingest_dir(path: str, background_tasks: BackgroundTasks = None):
    """
    Trigger ingestion of a directory already on disk.
    `path` is relative to repo root (e.g., 'data/book').
    This will start `ingest_multi_docs.py` in background.
    """
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {path}")
    out_index = f"backend/db/{p.name}_faiss.index"
    out_meta = f"backend/db/{p.name}_meta.jsonl"
    cmd = f"PYTHONPATH=. python backend/scripts/ingest_multi_docs.py --input {p} --out-index {out_index} --out-meta {out_meta} --batch 32 --chunk-tokens {settings.CHUNK_TOKENS} --overlap {settings.CHUNK_OVERLAP} --metric l2"
    _run_ingest_subprocess(cmd)
    return {"status": "accepted", "path": path, "ingest_cmd": cmd}


@APP.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if retrieve_service is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        results = retrieve_service.retrieve(req.query, top_k=req.top_k)
    except Exception as e:
        logger.exception("Retrieve failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"query": req.query, "results": results}


def assemble_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    instructions = (
        "You are a helpful assistant. Use the following extracted passages from the book to answer the question. "
        "Be concise and include citations in the form [chunk_id]. If you cannot find the answer in the passages, say so."
    )
    ctx_parts = []
    for i, c in enumerate(chunks):
        meta = c.get("meta", {})
        text = meta.get("text", "")
        cid = c.get("index", i)
        source = meta.get("pdf", meta.get("pid", "unknown"))
        header = f"[chunk {cid} | source: {source}]"
        ctx_parts.append(f"{header}\n{text}\n")
    context = "\n\n".join(ctx_parts)
    prompt = f"{instructions}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    return prompt


def call_ollama_generate(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Optional[str]:
    """
    Call Ollama to generate. Defensive: returns None on failure or when Ollama not configured.
    """
    if not settings.OLLAMA_MODEL or not ollama_client:
        return None
    try:
        # Ollama generate wrapper uses model at call time
        out = ollama_client.generate(model=settings.OLLAMA_MODEL, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        return out
    except Exception as e:
        logger.exception("Ollama generate failed: %s", e)
        return None


@APP.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if retrieve_service is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    # retrieve (with reranker internal to RetrieveService if enabled)
    try:
        if retrieve_service.reranker_enabled and reranker_obj:
            initial_k = max(settings.RERANKER_INITIAL_K, req.top_k)
            candidates = retrieve_service.retrieve(req.query, top_k=initial_k)
            # reranker object may have different method names
            try:
                results = reranker_obj.rerank_results(req.query, candidates, top_k=req.top_k)
            except Exception:
                # fallback to retrieve top_k
                results = candidates[:req.top_k]
        else:
            results = retrieve_service.retrieve(req.query, top_k=req.top_k)
    except Exception as e:
        logger.exception("Retrieve step failed: %s", e)
        raise HTTPException(status_code=500, detail="Retrieval failed")

    # Assemble prompt
    prompt = assemble_prompt(req.query, results)

    # Try generate via Ollama
    generated = call_ollama_generate(prompt, max_tokens=req.max_tokens, temperature=req.temperature)

    if generated is None:
        answer = "No generator available (Ollama not configured). Returning prompt and retrieved passages.\n\n" + prompt
    else:
        answer = generated

    sources = [{"index": r.get("index"), "score": r.get("score"), "meta": r.get("meta")} for r in results]
    return {"query": req.query, "answer": answer, "prompt": prompt, "sources": sources}
