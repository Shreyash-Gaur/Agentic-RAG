"""
Writer Agent - responsible for generating responses based on retrieved context.
"""

from typing import List, Dict, Any
from core.logger import setup_logger
from core.exceptions import AgentError

logger = setup_logger(__name__)


class WriterAgent:
    """
    Agent responsible for writing responses based on retrieved context.
    """
    
    def __init__(self, llm_client):
        """
        Initialize the writer agent.
        
        Args:
            llm_client: LLM client instance (e.g., OllamaClient)
        """
        self.llm_client = llm_client
    
    def write(
        self,
        query: str,
        context: List[Dict[str, Any]],
        system_prompt: str = None
    ) -> str:
        """
        Generate a response based on query and context.
        
        Args:
            query: User query
            context: Retrieved context documents
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        try:
            logger.info(f"Writing response for query: {query}")
            
            # Format context
            context_text = self._format_context(context)
            
            # Build prompt
            prompt = self._build_prompt(query, context_text, system_prompt)
            
            # Generate response
            response = self.llm_client.generate(prompt)
            
            logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error during writing: {e}")
            raise AgentError(f"Writing failed: {e}")
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context documents into a single string."""
        formatted = []
        for i, doc in enumerate(context, 1):
            content = doc.get('content', doc.get('text', ''))
            source = doc.get('source', doc.get('metadata', {}).get('source', 'Unknown'))
            formatted.append(f"[Document {i} - Source: {source}]\n{content}\n")
        return "\n".join(formatted)
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        system_prompt: str = None
    ) -> str:
        """Build the prompt for the LLM."""
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the context doesn't contain
enough information, say so."""
        
        prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""
        return prompt

