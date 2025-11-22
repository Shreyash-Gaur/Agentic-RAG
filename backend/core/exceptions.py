"""
Custom exceptions for the Agentic RAG system.
"""


class AgenticRAGException(Exception):
    """Base exception for Agentic RAG system."""
    pass


class RetrievalError(AgenticRAGException):
    """Error during document retrieval."""
    pass


class EmbeddingError(AgenticRAGException):
    """Error during embedding generation."""
    pass


class LLMError(AgenticRAGException):
    """Error during LLM interaction."""
    pass


class AgentError(AgenticRAGException):
    """Error during agent execution."""
    pass


class IngestionError(AgenticRAGException):
    """Error during document ingestion."""
    pass


class ConfigurationError(AgenticRAGException):
    """Error in configuration."""
    pass

