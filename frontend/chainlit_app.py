"""
Chainlit frontend application for interactive Agentic RAG testing.
"""

import chainlit as cl
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.config import settings
from core.logger import setup_logger
from tools.ollama_client import OllamaClient
from tools.embedder import Embedder
from tools.retriever_faiss import FAISSRetriever
from tools.retriever_chroma import ChromaRetriever
from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent
from services.rag_service import RAGService
from services.memory_service import MemoryService
from workflows.agentic_rag_flow import AgenticRAGFlow

logger = setup_logger(__name__)

# Initialize components (lazy loading)
_rag_flow = None


def get_rag_flow():
    """Initialize and return RAG flow (singleton)."""
    global _rag_flow
    if _rag_flow is None:
        # Initialize components
        embedder = Embedder()
        
        if settings.VECTOR_STORE_TYPE == "faiss":
            retriever = FAISSRetriever(embedder=embedder)
            retriever.load_index()
        else:
            retriever = ChromaRetriever(embedder=embedder)
        
        researcher = ResearcherAgent(retriever)
        writer = WriterAgent(OllamaClient())
        rag_service = RAGService(researcher, writer)
        memory_service = MemoryService()
        
        _rag_flow = AgenticRAGFlow(rag_service, memory_service)
    
    return _rag_flow


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    await cl.Message(
        content="Welcome to Agentic RAG! Ask me anything based on the ingested documents."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    try:
        rag_flow = get_rag_flow()
        
        # Generate conversation ID from session
        conversation_id = cl.user_session.get("id", "default")
        
        # Process query
        result = rag_flow.process_query(
            query=message.content,
            conversation_id=conversation_id,
            top_k=5,
            use_memory=True,
            iterative=False
        )
        
        # Send response
        response = f"{result['answer']}\n\n*Sources: {result['num_sources']} documents retrieved*"
        await cl.Message(content=response).send()
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await cl.Message(
            content=f"Sorry, an error occurred: {str(e)}"
        ).send()


if __name__ == "__main__":
    # Run with: chainlit run chainlit_app.py
    pass


