"""
Pydantic request models for API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., description="User query")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")


class IterativeQueryRequest(BaseModel):
    """Request model for iterative RAG query."""
    query: str = Field(..., description="User query")
    max_iterations: int = Field(3, ge=1, le=10, description="Maximum refinement iterations")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents per retrieval")


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    file_path: str = Field(..., description="Path to document file")
    file_type: str = Field("pdf", description="Type of file (pdf, txt, etc.)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional document metadata")


class BatchIngestRequest(BaseModel):
    """Request model for batch document ingestion."""
    file_paths: List[str] = Field(..., description="List of file paths")
    file_type: str = Field("pdf", description="Type of files")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="Optional metadata per document")

