# backend/main.py
import os
import sys
import time
import shlex
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from dotenv import load_dotenv

# --- Load Environment ---
load_dotenv()

# --- Internal Modules ---
from backend.core.config import settings
from backend.core.logger import get_logger

# Services
from backend.services.memory_service import MemoryService
from backend.services.embed_cache_service import EmbedCacheService
from backend.services.retrieve_service import RetrieveService

# Tools
from backend.tools.ollama_client import OllamaClient

# Agents
from backend.agents.researcher_agent import ResearcherAgent
from backend.agents.writer_agent import WriterAgent
from backend.agents.rag_agent import RAGAgent

# Models (Schema)
from backend.models.request_models import (
    QueryRequest, 
    RetrieveRequest
)
from backend.models.response_models import (
    QueryResponse, 
    RetrieveResponse, 
    DocumentResult
)

# --- Logging ---
logger = get_logger("agentic-rag")

# --- FastAPI App ---
APP = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# CORS
if settings.CORS:
    from fastapi.middleware.cors import CORSMiddleware
    APP.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- Globals (Singletons) ---
memory_service: Optional[MemoryService] = None
embed_cache: Optional[EmbedCacheService] = None
retrieve_service: Optional[RetrieveService] = None
ollama_client: Optional[OllamaClient] = None

# Agents
researcher_agent: Optional[ResearcherAgent] = None
writer_agent: Optional[WriterAgent] = None
rag_agent: Optional[RAGAgent] = None


@APP.on_event("startup")
def startup_event():
    """
    Initialize all services and agents on startup.
    """
    global memory_service, embed_cache, retrieve_service, ollama_client
    global researcher_agent, writer_agent, rag_agent

    logger.info("Starting Agentic-RAG service...")

    # 1. Memory Service
    try:
        memory_service = MemoryService(
            max_history=settings.MEMORY_MAX_TURNS,
            use_sqlite=True,
            db_path=settings.MEMORY_DB_PATH,
            preload=False,
        )
        logger.info(f"MemoryService initialized: {settings.MEMORY_DB_PATH}")
    except Exception as e:
        logger.exception(f"Failed to init MemoryService: {e}")

    # 2. Embed Cache
    try:
        embed_cache = EmbedCacheService(db_path=settings.EMBEDDING_CACHE_DB)
        logger.info(f"EmbedCacheService initialized: {settings.EMBEDDING_CACHE_DB}")
    except Exception as e:
        logger.exception(f"Failed to init EmbedCacheService: {e}")

    # 3. Reranker (Lazy loaded inside RetrieveService if passed, but we init object here)
    reranker_obj = None
    if settings.RERANKER_ENABLED:
        try:
            from backend.tools.reranker import CrossEncoderReranker
            reranker_obj = CrossEncoderReranker(model_name=settings.RERANKER_MODEL)
            logger.info(f"Reranker loaded: {settings.RERANKER_MODEL}")
        except Exception as e:
            logger.exception(f"Failed to load reranker: {e}")

    # 4. Ollama Client
    try:
        ollama_client = OllamaClient(base_url=settings.OLLAMA_BASE_URL)
        logger.info(f"Ollama client ready at {settings.OLLAMA_BASE_URL}")
    except Exception as e:
        logger.exception(f"Failed to init OllamaClient: {e}")

    # 5. Retrieve Service
    try:
        retrieve_service = RetrieveService(
            index_path=settings.FAISS_INDEX_PATH,
            meta_path=settings.FAISS_META_PATH,
            embed_cache=embed_cache,
            embedder=None,  # Uses default Embedder internally
            reranker_obj=reranker_obj,
            reranker_enabled=bool(reranker_obj),
        )
        logger.info(f"RetrieveService initialized (index={settings.FAISS_INDEX_PATH})")
    except Exception as e:
        logger.exception(f"Failed to init RetrieveService: {e}")

    # 6. Agents Wiring
    if retrieve_service:
        researcher_agent = ResearcherAgent(retriever_service=retrieve_service)
    
    if ollama_client:
        writer_agent = WriterAgent(
            ollama_client=ollama_client, 
            model=settings.OLLAMA_MODEL
        )

    if researcher_agent and writer_agent:
        rag_agent = RAGAgent(
            researcher=researcher_agent,
            writer=writer_agent,
            memory=memory_service
        )
        logger.info("RAGAgent successfully wired.")
    else:
        logger.warning("RAGAgent could not be wired (missing dependencies).")


@APP.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down Agentic-RAG service...")
    for svc in [retrieve_service, embed_cache, memory_service]:
        if svc and hasattr(svc, "close"):
            try:
                svc.close()
            except Exception:
                pass


@APP.get("/health")
def health():
    return {
        "status": "ok",
        "retriever": bool(retrieve_service),
        "rag_agent": bool(rag_agent),
        "ollama": bool(ollama_client)
    }


# ------------------------------
# Endpoints
# ------------------------------

@APP.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest):
    """
    Direct retrieval endpoint (bypasses Agentic reasoning).
    """
    if not retrieve_service:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        # Retrieve raw dicts
        raw_results = retrieve_service.retrieve(req.query, top_k=req.top_k)
        
        # Convert to Pydantic DocumentResult
        doc_results = []
        for r in raw_results:
            meta = r.get("meta", {})
            doc = DocumentResult(
                text=meta.get("text", ""),
                score=r.get("score", 0.0),
                metadata=meta,
                source=meta.get("pdf") or meta.get("doc_name") or "unknown",
                chunk_id=r.get("index")
            )
            doc_results.append(doc)

        return RetrieveResponse(
            query=req.query,
            results=doc_results,
            num_results=len(doc_results)
        )
    except Exception as e:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=500, detail=str(e))


@APP.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    Main RAG endpoint.
    Uses RAGAgent to orchestrate research -> writing.
    Supports conversation history if 'conversation_id' is provided.
    """
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")

    try:
        # Decide between stateless query or stateful chat
        if req.conversation_id and req.conversation_id != "default":
            # Statefull chat
            output = rag_agent.chat(
                conv_id=req.conversation_id,
                query=req.query,
                top_k=req.top_k,
                max_tokens=req.max_tokens,
                temperature=req.temperature
            )
        else:
            # Stateless query
            output = rag_agent.query(
                query=req.query,
                top_k=req.top_k,
                max_tokens=req.max_tokens,
                temperature=req.temperature
            )

        # Map 'sources' (list of dicts) to List[DocumentResult] for the response model
        sources_list = output.get("sources", [])
        mapped_sources = []
        for s in sources_list:
            meta = s.get("meta", {})
            doc = DocumentResult(
                text=meta.get("text", ""),
                score=s.get("score", 0.0),
                metadata=meta,
                source=meta.get("pdf", "unknown"),
                chunk_id=s.get("index")
            )
            mapped_sources.append(doc)

        return QueryResponse(
            query=req.query,
            answer=output.get("answer") or "No answer generated.",
            sources=mapped_sources,
            num_sources=len(mapped_sources),
            prompt=output.get("prompt"),
            metadata=output.get("timings")
        )

    except Exception as e:
        logger.exception(f"Agent Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
# Ingestion Endpoints (Legacy / Direct)
# ------------------------------
INGEST_UPLOAD_DIR = Path("data/book")
INGEST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _run_ingest_subprocess(cmd_list: List[str], env: Optional[Dict[str, str]] = None, log_prefix: str = "ingest") -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    log_dir = repo_root / "logs" / "ingests"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd_str = " ".join(shlex.quote(p) for p in cmd_list)
    ts = int(time.time())
    out_log = log_dir / f"{log_prefix}-{ts}.out.log"
    err_log = log_dir / f"{log_prefix}-{ts}.err.log"

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    out_f = open(out_log, "ab")
    err_f = open(err_log, "ab")

    p = subprocess.Popen(cmd_list, env=full_env, stdout=out_f, stderr=err_f, start_new_session=True)

    return {"pid": p.pid, "out_log": str(out_log), "err_log": str(err_log), "cmd": cmd_str}

# backend/main.py (Partial update for the bottom section)

@APP.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    filename = Path(file.filename).name
    out_path = INGEST_UPLOAD_DIR / filename
    with out_path.open("wb") as fh:
        content = await file.read()
        fh.write(content)

    # Use the multi-doc ingestion script instead of the old single-book one
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "backend" / "scripts" / "ingest_multi_docs.py"
    
    cmd_list = [
        sys.executable,
        str(script_path),
        "--input", str(out_path.resolve()),  # Changed from --pdf to --input
        "--out-index", settings.FAISS_INDEX_PATH, # Explicitly use config paths
        "--out-meta", settings.FAISS_META_PATH,
        "--chunk-tokens", str(settings.CHUNK_TOKENS),
        "--overlap", str(settings.CHUNK_OVERLAP),
        "--batch", "64",
        "--append"  # Add to existing index rather than overwriting
    ]
    env = {"PYTHONPATH": str(repo_root)}
    info = _run_ingest_subprocess(cmd_list, env=env, log_prefix="upload_ingest")

    return {
        "status": "accepted",
        "filename": filename,
        "pid": info["pid"],
        "logs": info["out_log"]
    }