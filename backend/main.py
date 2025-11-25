# backend/main.py
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from dotenv import load_dotenv
load_dotenv()

# services
from services.memory_service import MemoryService
from services.retrieve_service import RetrieveService

# agents
from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent
from agents.rag_agent import RAGAgent

# after creating retriever_service, memory_service
researcher = ResearcherAgent(retriever_service)
writer = WriterAgent()  # uses Ollama via env if present
rag_agent = RAGAgent(researcher, writer, memory=memory_service)

# config / env
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() in ("1", "true", "yes")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_INITIAL_K = int(os.getenv("RERANKER_INITIAL_K", "20"))

BACKEND_DIR = Path(__file__).resolve().parent
DB_DIR = BACKEND_DIR / "db"
DEFAULT_INDEX = DB_DIR / "book_king_faiss.index"
DEFAULT_META = DB_DIR / "book_king_meta.jsonl"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
USE_OLLAMA = bool(OLLAMA_MODEL)

CORS_ORIGINS = [s.strip() for s in os.getenv("CORS_ORIGINS", "").split(",") if s.strip()]
APP = FastAPI(title="Agentic-RAG")

if CORS_ORIGINS:
    from fastapi.middleware.cors import CORSMiddleware
    APP.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

logger = logging.getLogger("agentic-rag")
logging.basicConfig(level=logging.INFO)

# runtime globals
memory_service: Optional[MemoryService] = None
retrieve_service: Optional[RetrieveService] = None


# Pydantic models
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
    prompt: str
    sources: list

class ChatRequest(BaseModel):
    conv_id: Optional[str] = "default"
    query: str
    top_k: int = 5

class ChatResponse(BaseModel):
    conv_id: str
    answer: str
    sources: list
    history: list

# startup wiring
@APP.on_event("startup")
def startup_event():
    global memory_service, retrieve_service, reranker

    idx_path = os.getenv("FAISS_INDEX_PATH", str(DEFAULT_INDEX))
    meta_path = os.getenv("FAISS_META_PATH", str(DEFAULT_META))

    # instantiate reranker object if needed (RAG/agents may use)
    reranker_obj = None
    if RERANKER_ENABLED:
        try:
            from tools.reranker import CrossEncoderReranker
            reranker_obj = CrossEncoderReranker(model_name=RERANKER_MODEL)
            logger.info("Reranker loaded: %s", RERANKER_MODEL)
        except Exception as e:
            logger.warning("Reranker init failed: %s", e)
            reranker_obj = None

    # Retrieve service
    try:
        retrieve_service = RetrieveService(
            index_path=idx_path,
            meta_path=meta_path,
            reranker_obj=reranker_obj,
            reranker_enabled=RERANKER_ENABLED,
        )
        logger.info("RetrieveService initialized")
    except Exception as e:
        logger.error("Failed to init RetrieveService: %s", e)
        retriever_service = None

    # Memory service (SQLite-backed)
    try:
        memory_service = MemoryService(
            max_history=int(os.getenv("MAX_MEMORY_TURNS", 20)),
            use_sqlite=os.getenv("MEMORY_SQLITE", "true").lower() in ("1","true","yes"),
            db_path=os.getenv("MEMORY_DB_PATH", str(DB_DIR / "memory_store.sqlite")),
        )
        logger.info("MemoryService initialized")
    except Exception as e:
        logger.error("Failed to init MemoryService: %s", e)
        memory_service = None

    # RAG Service - wire your researcher/writer agents into it
    try:
        # If your rag_service requires retriever/memory/ollama clients pass them here.
        rag_service = RAGService(retriever=retriever_service, memory=memory_service, ollama_base_url=OLLAMA_BASE_URL, ollama_model=OLLAMA_MODEL)
        logger.info("RAGService initialized")
    except Exception as e:
        logger.warning("RAGService not initialized fully: %s", e)
        rag_service = None

@APP.get("/health")
def health():
    return {
        "status": "ok",
        "retriever_loaded": retriever_service is not None,
        "memory_loaded": memory_service is not None,
        "rag_loaded": rag_service is not None,
        "ollama": USE_OLLAMA
    }

# Retrieval endpoint (plain)
@APP.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if retriever_service is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    results = retriever_service.retrieve(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}

# Query endpoint (RAG -> generate via rag_service or Ollama)
@APP.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if retriever_service is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    # fetch top-k passages (retriever handles rerank)
    results = retriever_service.retrieve(req.query, top_k=req.top_k)

    # build prompt
    prompt_parts = []
    for r in results:
        meta = r.get("meta", {})
        text = meta.get("text", "")
        cid = r.get("index")
        prompt_parts.append(f"[chunk {cid}]\n{text}\n")
    prompt = "Use the following context to answer the question:\n\n" + "\n".join(prompt_parts) + f"\n\nQUESTION: {req.query}\n\nAnswer:"

    # Prefer RAG service generation if available
    answer = None
    if rag_service is not None:
        try:
            gen = rag_service.generate_from_prompt(prompt, max_tokens=req.max_tokens, temperature=req.temperature)
            answer = gen
        except Exception as e:
            logger.warning("RAGService generation failed: %s", e)
            answer = None

    # fallback Ollama direct call
    if answer is None and USE_OLLAMA:
        try:
            url = f"{OLLAMA_BASE_URL}/api/models/{OLLAMA_MODEL}/generate"
            payload = {"prompt": prompt, "max_tokens": req.max_tokens, "temperature": req.temperature}
            r = requests.post(url, json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                answer = data.get("generated") or data.get("text") or json.dumps(data)
            else:
                logger.warning("Ollama generate error: %s", r.status_code)
        except Exception as e:
            logger.warning("Ollama call failed: %s", e)

    if answer is None:
        answer = "No generator available. Returning prompt and retrieved passages.\n\n" + prompt

    sources = [{"index": r["index"], "score": r["score"], "meta": r["meta"]} for r in results]
    return {"query": req.query, "answer": answer, "prompt": prompt, "sources": sources}

# Chat endpoint: uses RAGService if available and stores in memory
@APP.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not available")

    conv_id = req.conv_id or "default"
    # run rag query (RAGService should support a query() method returning answer and sources/context)
    out = rag_service.query(req.query, top_k=req.top_k)
    answer = out.get("answer") or out.get("generated") or out.get("text") or ""
    sources = out.get("sources", [])

    # persist to memory
    if memory_service is not None:
        memory_service.add_turn(conv_id, "user", req.query, meta={"sources": None})
        memory_service.add_turn(conv_id, "assistant", answer, meta={"sources": sources})

    history = memory_service.get_history(conv_id) if memory_service else []
    return {"conv_id": conv_id, "answer": answer, "sources": sources, "history": history}
