"""
Script for rebuilding the vector store index.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings
from core.logger import setup_logger
from tools.retriever_faiss import FAISSRetriever
from tools.retriever_chroma import ChromaRetriever

logger = setup_logger(__name__)


def rebuild_faiss_index():
    """Rebuild FAISS index from metadata."""
    logger.info("Rebuilding FAISS index...")
    
    # TODO: Implement index rebuilding logic
    # This would typically involve:
    # 1. Loading all documents from metadata store
    # 2. Regenerating embeddings
    # 3. Rebuilding the FAISS index
    
    logger.warning("FAISS index rebuild not yet implemented")
    logger.info("To rebuild, re-ingest all documents using ingest_data.py")


def rebuild_chroma_index():
    """Rebuild ChromaDB index."""
    logger.info("Rebuilding ChromaDB index...")
    
    # ChromaDB automatically persists, so rebuilding is typically
    # just a matter of re-ingesting documents
    
    logger.warning("ChromaDB index rebuild not yet implemented")
    logger.info("To rebuild, re-ingest all documents using ingest_data.py")


def main():
    """Main entry point for index rebuild script."""
    parser = argparse.ArgumentParser(description="Rebuild vector store index")
    parser.add_argument(
        "--vector-store",
        type=str,
        default=settings.VECTOR_STORE_TYPE,
        choices=["faiss", "chroma"],
        help="Vector store type"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists"
    )
    
    args = parser.parse_args()
    
    if args.vector_store == "faiss":
        rebuild_faiss_index()
    else:
        rebuild_chroma_index()


if __name__ == "__main__":
    main()

