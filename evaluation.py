"""
Evaluation module for measuring retrieval performance.
Includes metrics for precision, recall, and relevance scoring.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using various metrics.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = []
    
    @staticmethod
    def calculate_precision_at_k(retrieved_ids: List[Any], relevant_ids: List[Any], k: int) -> float:
        """
        Calculate Precision@K: fraction of retrieved documents that are relevant.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if not retrieved_ids or k == 0:
            return 0.0
        
        retrieved_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
        precision = relevant_retrieved / len(retrieved_k)
        
        return precision
    
    @staticmethod
    def calculate_recall_at_k(retrieved_ids: List[Any], relevant_ids: List[Any], k: int) -> float:
        """
        Calculate Recall@K: fraction of relevant documents that are retrieved.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
        recall = relevant_retrieved / len(relevant_ids)
        
        return recall
    
    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
        
        Returns:
            F1 score (harmonic mean of precision and recall)
        """
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def calculate_mrr(retrieved_ids: List[Any], relevant_ids: List[Any]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
        
        Returns:
            MRR score
        """
        relevant_set = set(relevant_ids)
        
        for idx, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / idx
        
        return 0.0
    
    @staticmethod
    def calculate_ndcg_at_k(retrieved_ids: List[Any], relevant_ids: List[Any], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        Simplified version assuming binary relevance.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        
        # Calculate DCG
        dcg = 0.0
        for idx, doc_id in enumerate(retrieved_k, 1):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(idx + 1)
        
        # Calculate ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg
    
    def evaluate_query(self, query: str, retrieved_docs: List[Any], 
                       relevant_ids: List[Any], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Evaluate a single query across multiple metrics.
        
        Args:
            query: Query string
            retrieved_docs: Retrieved documents with metadata
            relevant_ids: List of relevant document IDs for this query
            k_values: List of k values to evaluate
        
        Returns:
            Dictionary of evaluation metrics
        """
        retrieved_ids = [doc.metadata.get('id', i) for i, doc in enumerate(retrieved_docs)]
        
        metrics = {
            'query': query,
            'num_retrieved': len(retrieved_ids),
            'num_relevant': len(relevant_ids)
        }
        
        # Calculate metrics for each k
        for k in k_values:
            precision = self.calculate_precision_at_k(retrieved_ids, relevant_ids, k)
            recall = self.calculate_recall_at_k(retrieved_ids, relevant_ids, k)
            f1 = self.calculate_f1_score(precision, recall)
            ndcg = self.calculate_ndcg_at_k(retrieved_ids, relevant_ids, k)
            
            metrics[f'precision@{k}'] = precision
            metrics[f'recall@{k}'] = recall
            metrics[f'f1@{k}'] = f1
            metrics[f'ndcg@{k}'] = ndcg
        
        # Calculate MRR
        metrics['mrr'] = self.calculate_mrr(retrieved_ids, relevant_ids)
        
        return metrics
    
    def evaluate_retrieval_latency(self, retriever, query: str, num_runs: int = 5) -> Dict[str, float]:
        """
        Measure retrieval latency.
        
        Args:
            retriever: LangChain retriever instance
            query: Query string
            num_runs: Number of runs for averaging
        
        Returns:
            Dictionary with latency statistics
        """
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.time()
            _ = retriever.get_relevant_documents(query)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'std_latency_ms': np.std(latencies)
        }
    
    def print_evaluation_report(self, metrics_list: List[Dict[str, Any]]):
        """
        Print a formatted evaluation report.
        
        Args:
            metrics_list: List of metrics dictionaries from evaluate_query
        """
        print("\n" + "="*80)
        print("RETRIEVAL EVALUATION REPORT")
        print("="*80)
        
        # Calculate average metrics
        k_values = [1, 3, 5, 10]
        avg_metrics = {}
        
        for k in k_values:
            avg_metrics[f'precision@{k}'] = np.mean([m.get(f'precision@{k}', 0) for m in metrics_list])
            avg_metrics[f'recall@{k}'] = np.mean([m.get(f'recall@{k}', 0) for m in metrics_list])
            avg_metrics[f'f1@{k}'] = np.mean([m.get(f'f1@{k}', 0) for m in metrics_list])
            avg_metrics[f'ndcg@{k}'] = np.mean([m.get(f'ndcg@{k}', 0) for m in metrics_list])
        
        avg_metrics['mrr'] = np.mean([m.get('mrr', 0) for m in metrics_list])
        
        print(f"\nTotal Queries Evaluated: {len(metrics_list)}")
        print("\nAverage Metrics:")
        print("-" * 80)
        
        for k in k_values:
            print(f"\nK = {k}:")
            print(f"  Precision@{k}: {avg_metrics[f'precision@{k}']:.4f}")
            print(f"  Recall@{k}:    {avg_metrics[f'recall@{k}']:.4f}")
            print(f"  F1@{k}:        {avg_metrics[f'f1@{k}']:.4f}")
            print(f"  NDCG@{k}:      {avg_metrics[f'ndcg@{k}']:.4f}")
        
        print(f"\nMean Reciprocal Rank (MRR): {avg_metrics['mrr']:.4f}")
        print("\n" + "="*80)
