"""
PDF document ingestion utilities.
"""

from pathlib import Path
from typing import List, Dict, Any
from core.logger import setup_logger
from core.exceptions import IngestionError

logger = setup_logger(__name__)

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available. Install with: pip install PyPDF2")


class PDFIngester:
    """
    Utility for ingesting PDF documents.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize PDF ingester.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 is required for PDF ingestion")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise IngestionError(f"Failed to extract PDF text: {e}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        chunks = []
        words = text.split()
        
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Handle overlap
                overlap_words = int(self.chunk_overlap / 10)  # Approximate
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_length = sum(len(w) + 1 for w in current_chunk)
            
            current_chunk.append(word)
            current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def ingest(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Ingest a PDF file and return chunks with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of document chunks with metadata
        """
        text = self.extract_text(pdf_path)
        chunks = self.chunk_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "content": chunk,
                "metadata": {
                    "source": str(pdf_path),
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            })
        
        return documents

