"""
Memory service for conversation history and context management.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from core.logger import setup_logger

logger = setup_logger(__name__)


class MemoryService:
    """
    Service for managing conversation memory and context.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize memory service.
        
        Args:
            max_history: Maximum number of conversation turns to keep
        """
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_turn(
        self,
        conversation_id: str,
        query: str,
        answer: str,
        context: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add a conversation turn to memory.
        
        Args:
            conversation_id: Unique conversation identifier
            query: User query
            answer: System answer
            context: Optional retrieved context
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        turn = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "context": context or []
        }
        
        self.conversations[conversation_id].append(turn)
        
        # Trim history if needed
        if len(self.conversations[conversation_id]) > self.max_history:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history:]
        
        logger.debug(f"Added turn to conversation {conversation_id}")
    
    def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of conversation turns
        """
        return self.conversations.get(conversation_id, [])
    
    def get_context_summary(self, conversation_id: str, max_turns: int = 3) -> str:
        """
        Get a summary of recent conversation context.
        
        Args:
            conversation_id: Conversation identifier
            max_turns: Number of recent turns to include
            
        Returns:
            Formatted context summary string
        """
        history = self.get_history(conversation_id)
        recent = history[-max_turns:] if len(history) > max_turns else history
        
        summary_parts = []
        for turn in recent:
            summary_parts.append(f"Q: {turn['query']}")
            summary_parts.append(f"A: {turn['answer']}")
        
        return "\n".join(summary_parts)
    
    def clear_conversation(self, conversation_id: str):
        """Clear a conversation from memory."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")

