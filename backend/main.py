# backend/main.py
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
