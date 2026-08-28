"""
Measures whether the cross-encoder reranker actually improves retrieval quality,
using the same Precision@k / Recall@k / MRR metrics as the baseline-vs-docling
comparison.

Compares:
  - "No rerank": hybrid retriever's top 5 results, as-is
  - "With rerank": hybrid retriever's top 10 candidates, cross-encoder reranked
                   down to the top 5

Usage:
    python evaluation/evaluate_reranker.py
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation_data import evaluation_data
from retrieval_metrics import evaluate_retrieval
from rag.hybrid_retriever import get_hybrid_retriever
from services.answer import rerank_docs


class NoRerankRetriever:
    """Wraps the hybrid retriever's plain top-k, no reranking."""
    def __init__(self, k=5):
        self.retriever = get_hybrid_retriever(k=k)

    def invoke(self, query):
        return self.retriever.invoke(query)


class RerankedRetriever:
    """Pulls more candidates than needed, then cross-encoder reranks down to k."""
    def __init__(self, k=5, candidate_k=10):
        self.candidate_retriever = get_hybrid_retriever(k=candidate_k)
        self.k = k

    def invoke(self, query):
        candidates = self.candidate_retriever.invoke(query)
        return rerank_docs(query, candidates, top_k=self.k)


def print_summary(label: str, summary: dict):
    print(f"\n{label} (k={summary['k']}, {summary['num_questions']} questions)")
    print(f"  Mean Precision@{summary['k']}: {summary['mean_precision_at_k']:.3f}")
    print(f"  Mean Recall@{summary['k']}:    {summary['mean_recall_at_k']:.3f}")
    print(f"  MRR:                    {summary['mrr']:.3f}")


def main():
    k = 5

    no_rerank = NoRerankRetriever(k=k)
    with_rerank = RerankedRetriever(k=k, candidate_k=10)

    print("Running retrieval evaluation...")

    no_rerank_results = evaluate_retrieval(no_rerank, evaluation_data, k=k)
    reranked_results = evaluate_retrieval(with_rerank, evaluation_data, k=k)

    print_summary("No Rerank", no_rerank_results["summary"])
    print_summary("With Rerank", reranked_results["summary"])

    print("\nImprovement (Reranked - No Rerank):")
    a = no_rerank_results["summary"]
    b = reranked_results["summary"]
    print(f"  Precision@{k}: {b['mean_precision_at_k'] - a['mean_precision_at_k']:+.3f}")
    print(f"  Recall@{k}:    {b['mean_recall_at_k'] - a['mean_recall_at_k']:+.3f}")
    print(f"  MRR:           {b['mrr'] - a['mrr']:+.3f}")


if __name__ == "__main__":
    main()