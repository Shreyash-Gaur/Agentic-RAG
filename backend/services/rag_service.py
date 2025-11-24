"""
RAG (Retrieval-Augmented Generation) service.
"""

from typing import List, Dict, Any, Optional
from core.logger import setup_logger
from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent

logger = setup_logger(__name__)


class RAGService:
    """
    Main RAG service orchestrating retrieval and generation.
    """
    
    def __init__(
        self,
        researcher: ResearcherAgent,
        writer: WriterAgent
    ):
        """
        Initialize RAG service.
        
        Args:
            researcher: Researcher agent instance
            writer: Writer agent instance
        """
        self.researcher = researcher
        self.writer = writer
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query through RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            system_prompt: Optional system prompt for generation
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        logger.info(f"Processing RAG query: {query}")
        
        # Step 1: Research/Retrieve
        context = self.researcher.research(query, top_k=top_k)
        
        # Step 2: Generate response
        answer = self.writer.write(query, context, system_prompt)
        
        return {
            "query": query,
            "answer": answer,
            "context": context,
            "num_sources": len(context)
        }
    
    def query_with_iteration(
        self,
        query: str,
        max_iterations: int = 3,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Process query with iterative refinement.
        
        Args:
            query: User query
            max_iterations: Maximum refinement iterations
            top_k: Number of documents per retrieval
            
        Returns:
            Dictionary with final answer and iteration history
        """
        logger.info(f"Processing iterative RAG query: {query}")
        
        current_query = query
        all_context = []
        iteration_history = []
        
        for iteration in range(max_iterations):
            # Retrieve
            context = self.researcher.research(current_query, top_k=top_k)
            all_context.extend(context)
            
            # Generate intermediate answer
            answer = self.writer.write(current_query, context)
            
            iteration_history.append({
                "iteration": iteration + 1,
                "query": current_query,
                "answer": answer,
                "context_count": len(context)
            })
            
            # Check if we should continue (simplified - could use LLM to decide)
            if iteration < max_iterations - 1:
                # Refine query for next iteration
                current_query = self.researcher.refine_query(query, [c.get('text', '') for c in context])
        
        # Final answer with all context
        final_answer = self.writer.write(query, all_context)
        
        return {
            "query": query,
            "answer": final_answer,
            "iterations": iteration_history,
            "total_context": len(all_context)
        }


