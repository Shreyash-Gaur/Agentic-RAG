"""
Researcher Agent - responsible for information gathering and retrieval.
"""

from typing import List, Dict, Any
from core.logger import setup_logger
from core.exceptions import AgentError

logger = setup_logger(__name__)


class ResearcherAgent:
    """
    Agent responsible for researching and retrieving relevant information.
    """
    
    def __init__(self, retriever, web_search=None):
        """
        Initialize the researcher agent.
        
        Args:
            retriever: Vector store retriever instance
            web_search: Optional web search tool
        """
        self.retriever = retriever
        self.web_search = web_search
    
    def research(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Research a query and retrieve relevant documents.
        
        Args:
            query: Research query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        try:
            logger.info(f"Researching query: {query}")
            
            # Retrieve from vector store
            results = self.retriever.retrieve(query, top_k=top_k)
            
            # Optionally augment with web search
            if self.web_search:
                web_results = self.web_search.search(query)
                results.extend(web_results)
            
            logger.info(f"Retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during research: {e}")
            raise AgentError(f"Research failed: {e}")
    
    def refine_query(self, original_query: str, context: List[str]) -> str:
        """
        Refine the query based on context.
        
        Args:
            original_query: Original query
            context: Context from previous retrievals
            
        Returns:
            Refined query string
        """
        # TODO: Implement query refinement logic
        return original_query

