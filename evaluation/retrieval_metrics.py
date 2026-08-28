"""
Standard information-retrieval metrics for evaluating a retriever independent
of LLM answer generation. Each function takes:
  - retrieved_sources: ordered list of source filenames the retriever returned
  - relevant_sources: the set of filenames that are actually correct for the query
"""

from typing import List
import os

def precision_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    """Of the top-k retrieved docs, what fraction are actually relevant?"""
    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_sources)
    hits = sum(1 for src in top_k if src in relevant_set)
    return hits / len(top_k)


def recall_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    """Of all the relevant docs that exist, what fraction did we find in the top-k?"""
    if not relevant_sources:
        return 0.0
    top_k = set(retrieved_sources[:k])
    relevant_set = set(relevant_sources)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    """1 / (rank of the first relevant doc found), or 0 if none found.
    Rewards finding a relevant doc EARLY, not just eventually."""
    relevant_set = set(relevant_sources)
    for rank, src in enumerate(retrieved_sources, start=1):
        if src in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(retriever, eval_data: List[dict], k: int = 5) -> dict:
    """
    Runs precision@k, recall@k, and MRR across a full eval set for one retriever.
    eval_data: list of {"question": ..., "relevant_sources": [...]}
    Returns per-question results plus averaged final metrics.
    """
    results = []
    total_precision, total_recall, total_rr = 0.0, 0.0, 0.0

    for item in eval_data:
        question = item["question"]
        relevant_sources = item.get("relevant_sources", [])

        docs = retriever.invoke(question)
        retrieved_sources = [os.path.basename(d.metadata.get("source", "")) for d in docs]

        p = precision_at_k(retrieved_sources, relevant_sources, k)
        r = recall_at_k(retrieved_sources, relevant_sources, k)
        rr = reciprocal_rank(retrieved_sources, relevant_sources)

        total_precision += p
        total_recall += r
        total_rr += rr

        results.append({
            "question": question,
            "retrieved_sources": retrieved_sources,
            "relevant_sources": relevant_sources,
            "precision_at_k": p,
            "recall_at_k": r,
            "reciprocal_rank": rr,
        })

    n = len(eval_data)
    summary = {
        "mean_precision_at_k": total_precision / n if n else 0.0,
        "mean_recall_at_k": total_recall / n if n else 0.0,
        "mrr": total_rr / n if n else 0.0,
        "k": k,
        "num_questions": n,
    }

    return {"per_question": results, "summary": summary}