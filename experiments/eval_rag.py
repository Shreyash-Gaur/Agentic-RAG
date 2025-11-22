"""
Evaluation script for end-to-end RAG performance.
"""

import argparse
from pathlib import Path
import sys
import json
from typing import List, Dict, Any

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

logger = setup_logger(__name__)


def evaluate_rag(
    queries: List[str],
    ground_truth_answers: List[str],
    rag_service: RAGService,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Evaluate RAG system performance.
    
    Args:
        queries: List of query strings
        ground_truth_answers: List of ground truth answers
        rag_service: RAG service instance
        top_k: Number of documents to retrieve
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = []
    
    for query, gt_answer in zip(queries, ground_truth_answers):
        try:
            # Get RAG response
            result = rag_service.query(query, top_k=top_k)
            
            results.append({
                "query": query,
                "predicted_answer": result["answer"],
                "ground_truth": gt_answer,
                "num_sources": result["num_sources"]
            })
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            results.append({
                "query": query,
                "error": str(e)
            })
    
    return {
        "results": results,
        "num_queries": len(queries),
        "num_successful": len([r for r in results if "error" not in r])
    }


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument(
        "--eval-file",
        type=str,
        required=True,
        help="JSON file with queries and ground truth answers"
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default=settings.VECTOR_STORE_TYPE,
        choices=["faiss", "chroma"],
        help="Vector store type"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    
    args = parser.parse_args()
    
    # Load evaluation data
    with open(args.eval_file, 'r') as f:
        data = json.load(f)
    
    queries = data.get("queries", [])
    ground_truth = data.get("answers", [])
    
    if len(queries) != len(ground_truth):
        logger.error("Queries and answers must have same length")
        return
    
    # Initialize RAG service
    embedder = Embedder()
    if args.vector_store == "faiss":
        retriever = FAISSRetriever(embedder=embedder)
        retriever.load_index()
    else:
        retriever = ChromaRetriever(embedder=embedder)
    
    researcher = ResearcherAgent(retriever)
    writer = WriterAgent(OllamaClient())
    rag_service = RAGService(researcher, writer)
    
    # Evaluate
    logger.info(f"Evaluating {len(queries)} queries...")
    metrics = evaluate_rag(queries, ground_truth, rag_service, args.top_k)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n=== RAG Evaluation Results ===")
    print(f"Total queries: {metrics['num_queries']}")
    print(f"Successful: {metrics['num_successful']}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

