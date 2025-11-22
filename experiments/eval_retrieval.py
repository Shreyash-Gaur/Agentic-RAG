"""
Evaluation script for retrieval performance.
"""

import argparse
from pathlib import Path
import sys
import json
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.config import settings
from core.logger import setup_logger
from tools.embedder import Embedder
from tools.retriever_faiss import FAISSRetriever
from tools.retriever_chroma import ChromaRetriever

logger = setup_logger(__name__)


def evaluate_retrieval(
    queries: List[str],
    ground_truth: List[List[str]],
    retriever,
    top_k: int = 5
) -> Dict[str, float]:
    """
    Evaluate retrieval performance.
    
    Args:
        queries: List of query strings
        ground_truth: List of lists of relevant document IDs
        retriever: Retriever instance
        top_k: Number of documents to retrieve
        
    Returns:
        Dictionary with evaluation metrics
    """
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    for query, gt_ids in zip(queries, ground_truth):
        # Retrieve documents
        results = retriever.retrieve(query, top_k=top_k)
        retrieved_ids = [r.get("id") for r in results if r.get("id")]
        
        # Calculate metrics
        if retrieved_ids and gt_ids:
            relevant_retrieved = len(set(retrieved_ids) & set(gt_ids))
            precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0
            recall = relevant_retrieved / len(gt_ids) if gt_ids else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
    
    n = len(queries)
    return {
        "precision": total_precision / n if n > 0 else 0.0,
        "recall": total_recall / n if n > 0 else 0.0,
        "f1": total_f1 / n if n > 0 else 0.0,
        "num_queries": n
    }


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance")
    parser.add_argument(
        "--queries-file",
        type=str,
        required=True,
        help="JSON file with queries and ground truth"
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default=settings.VECTOR_STORE_TYPE,
        choices=["faiss", "chroma"],
        help="Vector store type"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    
    args = parser.parse_args()
    
    # Load queries and ground truth
    with open(args.queries_file, 'r') as f:
        data = json.load(f)
    
    queries = data.get("queries", [])
    ground_truth = data.get("ground_truth", [])
    
    if len(queries) != len(ground_truth):
        logger.error("Queries and ground truth must have same length")
        return
    
    # Initialize retriever
    embedder = Embedder()
    if args.vector_store == "faiss":
        retriever = FAISSRetriever(embedder=embedder)
        retriever.load_index()
    else:
        retriever = ChromaRetriever(embedder=embedder)
    
    # Evaluate
    metrics = evaluate_retrieval(queries, ground_truth, retriever, args.top_k)
    
    # Print results
    print("\n=== Retrieval Evaluation Results ===")
    print(f"Precision@{args.top_k}: {metrics['precision']:.4f}")
    print(f"Recall@{args.top_k}: {metrics['recall']:.4f}")
    print(f"F1@{args.top_k}: {metrics['f1']:.4f}")
    print(f"Number of queries: {metrics['num_queries']}")


if __name__ == "__main__":
    main()

