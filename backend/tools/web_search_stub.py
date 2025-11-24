"""
Web search tool stub - placeholder for actual web search integration.
"""

from typing import List, Dict, Any
from core.logger import setup_logger

logger = setup_logger(__name__)


class WebSearchStub:
    """
    Stub implementation for web search.
    Replace with actual web search API (e.g., Tavily, Serper, etc.)
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize web search stub.
        
        Args:
            api_key: API key for web search service (not used in stub)
        """
        self.api_key = api_key
        logger.info("WebSearchStub initialized (stub implementation)")
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for a query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results with content and metadata
        """
        logger.warning(f"WebSearchStub.search called with query: {query} (stub - no actual search performed)")
        
        # Return empty results as stub
        return []
        
        # Example of what actual implementation might return:
        # return [
        #     {
        #         "content": "Search result content...",
        #         "source": "https://example.com",
        #         "title": "Example Result",
        #         "metadata": {"url": "https://example.com"}
        #     }
        # ]


