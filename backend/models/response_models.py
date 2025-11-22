"""
Pydantic response models for API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class DocumentResult(BaseModel):
    """Model for a retrieved document."""
    text: str = Field(..., description="Document text")
    content: str = Field(..., description="Document content (alias for text)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    score: float = Field(..., description="Retrieval score")
    source: Optional[str] = Field(None, description="Document source")


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    context: List[DocumentResult] = Field(..., description="Retrieved context documents")
    num_sources: int = Field(..., description="Number of sources used")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class IterativeQueryResponse(BaseModel):
    """Response model for iterative RAG query."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Final generated answer")
    iterations: List[Dict[str, Any]] = Field(..., description="Iteration history")
    total_context: int = Field(..., description="Total context documents retrieved")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool = Field(..., description="Whether ingestion was successful")
    file_path: str = Field(..., description="Path to ingested file")
    num_chunks: int = Field(..., description="Number of chunks created")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    version: Optional[str] = Field(None, description="API version")

