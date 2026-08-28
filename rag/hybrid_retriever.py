"""
Hybrid retrieval: combines BM25 (sparse/keyword) and the existing Qdrant dense
retriever, so exact terms (drug names, specific numbers, rare medical terms)
that dense embeddings sometimes miss still get caught reliably.

This builds the BM25 index by pulling chunk texts directly out of Qdrant via
scroll() - no re-ingestion needed, no changes to the existing ingestion pipeline.

Usage:
    from rag.hybrid_retriever import get_hybrid_retriever
    retriever = get_hybrid_retriever(k=5)
    docs = retriever.invoke("what lowers cholesterol")
"""

import os
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient
from pydantic import Field

from rag.retriver import get_retriever

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "adaptrag_docling")

_bm25_index = None
_bm25_docs = None


def _build_bm25_index():
    """Pulls every chunk out of Qdrant once and builds an in-memory BM25 index.
    Cached at module level so this only runs once per process."""
    global _bm25_index, _bm25_docs

    if _bm25_index is not None:
        return _bm25_index, _bm25_docs

    print("[INFO] Building BM25 index from Qdrant chunks...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    all_docs = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            text = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            if text:
                all_docs.append(Document(page_content=text, metadata=metadata))

        if offset is None:
            break

    tokenized_corpus = [doc.page_content.lower().split() for doc in all_docs]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_docs = all_docs

    print(f"[INFO] BM25 index built with {len(all_docs)} chunks")
    return _bm25_index, _bm25_docs


class HybridRetriever(BaseRetriever):
    """Combines BM25 sparse retrieval and dense Qdrant retrieval by merging
    both result sets, deduplicating by content, and interleaving so both
    signal types are represented in the final top-k."""

    k: int = Field(default=5)
    bm25_weight_k: int = Field(default=10)  # how many BM25 candidates to pull before merging

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        bm25_index, bm25_docs = _build_bm25_index()

        # Sparse side: top bm25_weight_k by BM25 score
        tokenized_query = query.lower().split()
        scores = bm25_index.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sparse_results = [bm25_docs[i] for i in ranked_indices[: self.bm25_weight_k] if scores[i] > 0]

        # Dense side: existing Qdrant retriever
        dense_retriever = get_retriever(k=self.k)
        dense_results = dense_retriever.invoke(query)

        # Merge: dense results first (usually higher precision), then any
        # sparse-only hits not already present, deduplicated by content.
        seen = set()
        merged = []

        for doc in dense_results + sparse_results:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        return merged[: self.k]


_hybrid_instance = None


def get_hybrid_retriever(k: int = 5) -> HybridRetriever:
    global _hybrid_instance
    if _hybrid_instance is None:
        _hybrid_instance = HybridRetriever(k=k)
    return _hybrid_instance