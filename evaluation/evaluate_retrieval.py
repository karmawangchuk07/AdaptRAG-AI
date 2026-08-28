"""
Retrieval-only evaluation: measures Precision@k, Recall@k, and MRR for the
baseline and docling retrievers, without calling the LLM at all. Much faster
than evaluate.py since there's no answer generation - just retrieval.

Usage:
    python evaluation/evaluate_retrieval.py
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation_data import evaluation_data
from retrieval_metrics import evaluate_retrieval
from rag.retriver import get_retriever
from rag.ingest import load_baseline_db


def print_summary(label: str, summary: dict):
    print(f"\n{label} (k={summary['k']}, {summary['num_questions']} questions)")
    print(f"  Mean Precision@{summary['k']}: {summary['mean_precision_at_k']:.3f}")
    print(f"  Mean Recall@{summary['k']}:    {summary['mean_recall_at_k']:.3f}")
    print(f"  MRR:                    {summary['mrr']:.3f}")


def main():
    k = 5

    baseline_vectordb = load_baseline_db()
    baseline_retriever = baseline_vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    docling_retriever = get_retriever(k=k)

    print("Running retrieval evaluation...")

    baseline_results = evaluate_retrieval(baseline_retriever, evaluation_data, k=k)
    docling_results = evaluate_retrieval(docling_retriever, evaluation_data, k=k)

    print_summary("Baseline", baseline_results["summary"])
    print_summary("Docling", docling_results["summary"])

    print("\nImprovement (Docling - Baseline):")
    b = baseline_results["summary"]
    d = docling_results["summary"]
    print(f"  Precision@{k}: {d['mean_precision_at_k'] - b['mean_precision_at_k']:+.3f}")
    print(f"  Recall@{k}:    {d['mean_recall_at_k'] - b['mean_recall_at_k']:+.3f}")
    print(f"  MRR:           {d['mrr'] - b['mrr']:+.3f}")


if __name__ == "__main__":
    main()