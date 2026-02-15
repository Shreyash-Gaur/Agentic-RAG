# backend/main.py
import os
import sys
import time
import shlex
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# --- Core Imports ---
from backend.core.config import settings
from backend.core.logger import setup_logging
from backend.core.exceptions import AgenticRAGException

# --- Services ---
from backend.services.retrieve_service import RetrieveService
from backend.services.embed_cache_service import EmbedCacheService
from backend.services.memory_service import MemoryService
from backend.services.semantic_cache_service import SemanticCacheService

# --- Tools ---
from backend.tools.reranker import Reranker

# --- Agents ---
from backend.agents.graph_agent import GraphRAGAgent

# --- Models ---
from backend.models.request_models import QueryRequest, RetrieveRequest
from backend.models.response_models import QueryResponse, RetrieveResponse, DocumentResult

setup_logging()
logger = logging.getLogger("agentic-rag.api")

# --- Global Services ---
retrieve_service: Optional[RetrieveService] = None
embed_cache: Optional[EmbedCacheService] = None
memory_service: Optional[MemoryService] = None
semantic_cache: Optional[SemanticCacheService] = None
rag_agent: Optional[GraphRAGAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifecycle manager to replace @on_event('startup'/'shutdown')"""
    global memory_service, embed_cache, retrieve_service, semantic_cache, rag_agent
    
    logger.info("Starting Agentic-RAG service...")

    # 1. Memory Service
    try:
        memory_service = MemoryService()
        logger.info("MemoryService initialized.")
    except Exception as e:
        logger.exception(f"Failed to init MemoryService: {e}")

    # 2. Embed Cache
    try:
        embed_cache = EmbedCacheService()
        logger.info("EmbedCacheService initialized.")
    except Exception as e:
        logger.exception(f"Failed to init EmbedCacheService: {e}")

    # 3. Reranker
    reranker_obj = None
    if settings.RERANKER_ENABLED:
        try:
            reranker_obj = Reranker()
            logger.info(f"Reranker loaded: {settings.RERANKER_MODEL}")
        except Exception as e:
            logger.exception(f"Failed to load reranker: {e}")

    # 4. Semantic Cache
    try:
        semantic_cache = SemanticCacheService()
        logger.info("SemanticCacheService initialized.")
    except Exception as e:
        logger.exception(f"Failed to init SemanticCacheService: {e}")

    # 5. Retrieve Service
    try:
        retrieve_service = RetrieveService(
            embed_cache=embed_cache,
            reranker_obj=reranker_obj,
            reranker_enabled=settings.RERANKER_ENABLED,
        )
        logger.info(f"RetrieveService initialized (index={settings.FAISS_INDEX_PATH})")
    except Exception as e:
        logger.exception(f"Failed to init RetrieveService: {e}")

    # 6. Initialize Graph Agent
    if retrieve_service:
        try:
            rag_agent = GraphRAGAgent(
                retrieve_service=retrieve_service,
                model_name=settings.OLLAMA_MODEL
            )
            logger.info("GraphRAGAgent successfully initialized.")
        except Exception as e:
            logger.exception(f"Failed to init GraphRAGAgent: {e}")
    else:
        logger.warning("RetrieveService missing. GraphRAGAgent not initialized.")

    yield # --- App is running ---

    # Shutdown
    logger.info("Shutting down Agentic-RAG service...")
    for svc in [retrieve_service, embed_cache, memory_service]:
        if svc and hasattr(svc, "close"):
            try:
                svc.close()
            except Exception:
                pass


APP = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION, lifespan=lifespan)

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@APP.get("/health")
def health():
    return {
        "status": "ok",
        "retriever": bool(retrieve_service),
        "rag_agent": bool(rag_agent),
        "semantic_cache": bool(semantic_cache),
        "memory_service": bool(memory_service)
    }


@APP.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest):
    if not retrieve_service:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    try:
        raw_results = retrieve_service.retrieve(req.query, top_k=req.top_k)
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
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")
    try:
        session_id = req.conversation_id or "default"
        user_query = req.query
        
        # 1. Semantic Cache Check (Skip LLM if similar question was asked)
        if semantic_cache:
            cached_answer = semantic_cache.check_cache(user_query)
            if cached_answer:
                logger.info("Serving response from Semantic Cache.")
                if memory_service:
                    memory_service.add_turn(session_id, user_query, cached_answer)
                return QueryResponse(
                    query=user_query,
                    answer=cached_answer,
                    sources=[],
                    num_sources=0,
                    prompt="",
                    metadata={"source": "semantic_cache", "steps": ["cache_hit"]}
                )

        # 2. Retrieve Memory Context
        chat_history = ""
        if memory_service:
            chat_history = memory_service.get_context(session_id)

        # 3. Execute GraphRAGAgent
        mode = "detailed" if req.max_tokens > 512 else "concise"
        output = rag_agent.query(
            query=user_query, 
            mode=mode, 
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            chat_history=chat_history
        )
        
        ai_answer = output.get("answer", "No answer generated.")
        
        # 4. Save to Memory & Semantic Cache
        if memory_service:
            memory_service.add_turn(session_id, user_query, ai_answer)
        if semantic_cache:
            semantic_cache.add_new_turn(user_query, ai_answer)

        return QueryResponse(
            query=user_query,
            answer=ai_answer,
            sources=[], 
            num_sources=0,
            prompt="",
            metadata=output.get("metadata", {})
        )
    except Exception as e:
        logger.exception(f"Agent Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Upload & Ingestion Logic ---

INGEST_UPLOAD_DIR = Path(settings.WATCH_DIR)
INGEST_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _run_ingest_subprocess(cmd_list: List[str], env: Optional[Dict[str, str]] = None, log_prefix: str = "ingest") -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1] 
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
    if "PYTHONPATH" not in full_env:
        full_env["PYTHONPATH"] = str(repo_root)

    p = subprocess.Popen(cmd_list, env=full_env, stdout=out_f, stderr=err_f, start_new_session=True)
    return {"pid": p.pid, "out_log": str(out_log), "err_log": str(err_log), "cmd": cmd_str}

@APP.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    filename = Path(file.filename).name
    out_path = INGEST_UPLOAD_DIR / filename
    with out_path.open("wb") as fh:
        content = await file.read()
        fh.write(content)

    repo_root = Path(__file__).resolve().parents[1]
    
    # 1. Run standard ingestion
    script_path = repo_root / "backend" / "scripts" / "ingest_multi_docs.py"
    cmd_list = [
        sys.executable,
        str(script_path),
        "--input", str(out_path.resolve()), 
        "--out-index", settings.FAISS_INDEX_PATH,
        "--out-meta", settings.FAISS_META_PATH,
        "--chunk-tokens", str(settings.CHUNK_TOKENS),
        "--overlap", str(settings.CHUNK_OVERLAP),
        "--batch", str(settings.EMBEDDING_BATCH_SIZE),
        "--append" 
    ]
    env = {"PYTHONPATH": str(repo_root)}
    info = _run_ingest_subprocess(cmd_list, env=env, log_prefix="upload_ingest")
    
    # Note: If running ingest_vector_watch.py in the background, it will automatically catch this
    # uploaded file and run the SQLite sync automatically.

    return {
        "status": "accepted",
        "filename": filename,
        "pid": info["pid"],
        "logs": info["out_log"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:APP", host="0.0.0.0", port=8000, reload=settings.DEBUG)