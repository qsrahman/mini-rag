"""Retrieval: dense vector search, optionally fused with BM25 keyword search. No generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mini_rag.embeddings import Embedder
from mini_rag.models import RetrievedChunk
from mini_rag.reranker import LlmReranker
from mini_rag.vector_store import VectorStore

RRF_K = 60  # standard constant from the original RRF paper
CANDIDATE_FACTOR = 4  # each retriever contributes this many times top_k candidates...
MIN_CANDIDATES = 20  # ...but never fewer than this
RERANK_POOL = 10  # fused results handed to the reranker (more = slower, one LLM call each)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    """Merge ranked id lists by rank position (not score): sum of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: scores[item], reverse=True)  # stable on ties


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 5,
        mode: str = "dense",
        reranker: LlmReranker | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if mode not in ("dense", "hybrid"):
            raise ValueError(f"mode must be 'dense' or 'hybrid', got {mode!r}")
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._mode = mode
        self._reranker = reranker

    def retrieve(
        self, question: str, top_k: int | None = None, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        k = self._top_k if top_k is None else top_k
        pool = max(k, RERANK_POOL) if self._reranker else k
        results = self._search(question, pool, where)
        if self._reranker:
            return self._reranker.rerank(question, results)[:k]
        return results

    def _search(self, question: str, k: int, where: dict[str, Any] | None) -> list[RetrievedChunk]:
        embedding = self._embedder.embed([question])[0]
        if self._mode == "dense":
            return self._store.query(embedding, k, where)

        depth = max(k * CANDIDATE_FACTOR, MIN_CANDIDATES)
        dense = self._store.query(embedding, depth, where)
        keyword = self._store.keyword_query(question, depth, where)
        # Dense results go in last so their distance wins when both retrievers found a chunk.
        by_id = {r.chunk.id: r for r in (*keyword, *dense)}
        fused = reciprocal_rank_fusion(
            [
                [r.chunk.id for r in dense],
                [r.chunk.id for r in keyword],
            ]
        )
        return [by_id[cid] for cid in fused[:k]]
