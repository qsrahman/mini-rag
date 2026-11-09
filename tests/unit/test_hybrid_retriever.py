from unittest.mock import MagicMock

import pytest

from mini_rag.models import Chunk, RetrievedChunk
from mini_rag.retriever import Retriever, reciprocal_rank_fusion
from mini_rag.vector_store import chunk_id


def rc(name: str, distance: float | None = None) -> RetrievedChunk:
    meta = {"document_id": name, "content_hash": "h" * 12, "chunk_index": 0}
    return RetrievedChunk(Chunk(name, meta), distance)


def cid(name: str) -> str:
    return chunk_id(rc(name).chunk.metadata)


def test_rrf_rewards_agreement_between_rankings():
    # b is 2nd in both lists; c is 3rd and 1st. 1/63 + 1/61 narrowly beats 2/62.
    assert reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "d"]]) == ["c", "b", "a", "d"]
    assert reciprocal_rank_fusion([["a", "b"], ["b", "a"], ["b", "x"]])[0] == "b"


def test_rrf_single_ranking_and_empty():
    assert reciprocal_rank_fusion([["x", "y"]]) == ["x", "y"]
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_uses_rank_not_score_so_ties_are_stable():
    assert reciprocal_rank_fusion([["a"], ["b"]]) == ["a", "b"]


def make(embedder, dense, keyword, **kw):
    store = MagicMock()
    store.query.return_value = dense
    store.keyword_query.return_value = keyword
    return Retriever(embedder, store, mode="hybrid", **kw), store


def test_hybrid_fuses_and_truncates_to_top_k(embedder):
    retriever, _ = make(embedder, [rc("a", 0.1), rc("b", 0.2)], [rc("c"), rc("b")], top_k=2)
    results = retriever.retrieve("q")
    assert [chunk_id(r.chunk.metadata) for r in results] == [cid("b"), cid("a")]


def test_hybrid_keeps_dense_distance_when_both_found_a_chunk(embedder):
    retriever, _ = make(embedder, [rc("b", 0.25)], [rc("b")], top_k=1)
    assert retriever.retrieve("q")[0].distance == 0.25


def test_keyword_only_hits_have_no_distance(embedder):
    retriever, _ = make(embedder, [], [rc("k")], top_k=1)
    assert retriever.retrieve("q")[0].distance is None


def test_hybrid_asks_each_retriever_for_more_candidates_than_top_k(embedder):
    retriever, store = make(embedder, [], [], top_k=5)
    retriever.retrieve("q", where={"source": "a.md"})
    assert store.query.call_args.args[1] == 20
    assert store.keyword_query.call_args.args == ("q", 20, {"source": "a.md"})
    retriever.retrieve("q", top_k=10)
    assert store.keyword_query.call_args.args[1] == 40


def test_dense_mode_never_touches_keyword_search(embedder):
    store = MagicMock()
    store.query.return_value = []
    Retriever(embedder, store, top_k=3, mode="dense").retrieve("q")
    store.keyword_query.assert_not_called()


def test_invalid_mode_rejected(embedder):
    with pytest.raises(ValueError, match="mode"):
        Retriever(embedder, MagicMock(), mode="sparse")
