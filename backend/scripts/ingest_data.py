"""
Script for ingesting documents into the vector store.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings
from core.logger import setup_logger
from tools.embedder import Embedder
from tools.pdf_ingest import PDFIngester
from tools.retriever_faiss import FAISSRetriever
from tools.retriever_chroma import ChromaRetriever

logger = setup_logger(__name__)


def ingest_pdf(pdf_path: str, vector_store_type: str = "faiss"):
    """
    Ingest a PDF file into the vector store.
    
    Args:
        pdf_path: Path to PDF file
        vector_store_type: Type of vector store ("faiss" or "chroma")
    """
    logger.info(f"Ingesting PDF: {pdf_path}")
    
    # Initialize components
    embedder = Embedder()
    ingester = PDFIngester(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    # Initialize retriever
    if vector_store_type == "faiss":
        retriever = FAISSRetriever(embedder=embedder)
        retriever.load_index()
    else:
        retriever = ChromaRetriever(embedder=embedder)
    
    # Ingest PDF
    documents = ingester.ingest(pdf_path)
    
    # Generate embeddings
    texts = [doc["text"] for doc in documents]
    metadata = [doc["metadata"] for doc in documents]
    
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embedder.embed_batch(texts)
    
    # Add to vector store
    retriever.add_documents(texts, embeddings, metadata)
    
    # Save index
    if vector_store_type == "faiss":
        retriever.save_index()
    
    logger.info(f"Successfully ingested {len(documents)} chunks from {pdf_path}")


def main():
    """Main entry point for ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument("file_path", type=str, help="Path to document file")
    parser.add_argument(
        "--vector-store",
        type=str,
        default=settings.VECTOR_STORE_TYPE,
        choices=["faiss", "chroma"],
        help="Vector store type"
    )
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    if file_path.suffix.lower() == ".pdf":
        ingest_pdf(str(file_path), args.vector_store)
    else:
        logger.error(f"Unsupported file type: {file_path.suffix}")
        logger.info("Currently only PDF files are supported")


if __name__ == "__main__":
    main()

