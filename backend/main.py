"""
Main entry point for the Agentic RAG backend API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Agentic RAG API",
    description="Retrieval-Augmented Generation with Agentic Workflows",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Agentic RAG API is running"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


