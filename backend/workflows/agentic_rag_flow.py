"""
Agentic RAG workflow orchestrating the complete pipeline.
"""

from typing import Dict, Any, Optional
from core.logger import setup_logger
from services.rag_service import RAGService
from services.memory_service import MemoryService

logger = setup_logger(__name__)


class AgenticRAGFlow:
    """
    Main workflow orchestrating agentic RAG operations.
    """
    
    def __init__(
        self,
        rag_service: RAGService,
        memory_service: MemoryService
    ):
        """
        Initialize agentic RAG flow.
        
        Args:
            rag_service: RAG service instance
            memory_service: Memory service instance
        """
        self.rag_service = rag_service
        self.memory_service = memory_service
    
    def process_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
        use_memory: bool = True,
        iterative: bool = False,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Process a query through the agentic RAG flow.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID for context
            top_k: Number of documents to retrieve
            use_memory: Whether to use conversation memory
            iterative: Whether to use iterative refinement
            max_iterations: Maximum iterations if iterative=True
            
        Returns:
            Complete response with answer and metadata
        """
        logger.info(f"Processing query in agentic flow: {query}")
        
        # Get conversation context if available
        context_summary = None
        if use_memory and conversation_id:
            context_summary = self.memory_service.get_context_summary(conversation_id)
            if context_summary:
                # Augment query with context
                query = f"Previous conversation:\n{context_summary}\n\nCurrent question: {query}"
        
        # Process query
        if iterative:
            result = self.rag_service.query_with_iteration(
                query,
                max_iterations=max_iterations,
                top_k=top_k
            )
        else:
            result = self.rag_service.query(query, top_k=top_k)
        
        # Store in memory
        if use_memory and conversation_id:
            self.memory_service.add_turn(
                conversation_id,
                query,
                result["answer"],
                result.get("context", [])
            )
        
        return result
    
    def reset_conversation(self, conversation_id: str):
        """Reset a conversation."""
        self.memory_service.clear_conversation(conversation_id)
        logger.info(f"Reset conversation {conversation_id}")


