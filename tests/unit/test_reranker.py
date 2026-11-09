import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mini_rag.models import Chunk, RetrievedChunk
from mini_rag.ollama_utils import OllamaError
from mini_rag.reranker import LlmReranker
from mini_rag.retriever import Retriever


def rc(text: str) -> RetrievedChunk:
    return RetrievedChunk(Chunk(text, {"document_id": text, "content_hash": "h", "chunk_index": 0}))


def reply(yes: float | None, no: float | None = None):
    alts = []
    if yes is not None:
        alts.append(SimpleNamespace(token="Yes", logprob=math.log(yes)))
    if no is not None:
        alts.append(SimpleNamespace(token=" no", logprob=math.log(no)))
    return SimpleNamespace(logprobs=[SimpleNamespace(top_logprobs=alts)])


def reranker_scoring(by_document: dict[str, tuple[float | None, float | None]]):
    client = MagicMock()

    def chat(**kwargs):
        document = kwargs["messages"][1]["content"].split("<Document>: ")[1]
        return reply(*by_document[document])

    client.chat.side_effect = chat
    return LlmReranker("m", client=client), client


def test_candidates_are_ordered_by_yes_probability():
    reranker, client = reranker_scoring({"a": (0.2, 0.8), "b": (0.9, 0.1), "c": (0.5, 0.5)})
    result = reranker.rerank("q", [rc("a"), rc("b"), rc("c")])
    assert [r.chunk.text for r in result] == ["b", "c", "a"]
    kwargs = client.chat.call_args.kwargs
    assert kwargs["logprobs"] is True and kwargs["think"] is False
    assert kwargs["options"]["num_predict"] == 1


def test_ties_keep_the_incoming_order():
    reranker, _ = reranker_scoring({"a": (0.5, 0.5), "b": (0.5, 0.5), "c": (0.5, 0.5)})
    assert [r.chunk.text for r in reranker.rerank("q", [rc("a"), rc("b"), rc("c")])] == [
        "a",
        "b",
        "c",
    ]


def test_candidate_without_yes_or_no_scores_lowest():
    reranker, _ = reranker_scoring({"a": (None, None), "b": (0.1, 0.9)})
    assert [r.chunk.text for r in reranker.rerank("q", [rc("a"), rc("b")])] == ["b", "a"]


def test_missing_logprobs_is_a_clear_error():
    client = MagicMock()
    client.chat.return_value = SimpleNamespace(logprobs=None)
    with pytest.raises(OllamaError, match="logprobs"):
        LlmReranker("m", client=client).rerank("q", [rc("a")])


def test_retriever_reranks_a_pool_and_truncates_to_top_k(embedder):
    store = MagicMock()
    store.query.return_value = [
        rc(n) for n in "abcdefghijkl"
    ]  # dense mode returns what it is given
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, cands: list(reversed(cands))
    retriever = Retriever(embedder, store, top_k=3, mode="dense", reranker=reranker)

    results = retriever.retrieve("q")

    assert store.query.call_args.args[1] == 10  # pool is max(top_k, 10)
    assert reranker.rerank.call_args.args[0] == "q"
    assert [r.chunk.text for r in results] == ["l", "k", "j"]


def test_retriever_without_reranker_is_unchanged(embedder):
    store = MagicMock()
    store.query.return_value = [rc("a")]
    assert Retriever(embedder, store, top_k=3, mode="dense").retrieve("q") == [rc("a")]
    assert store.query.call_args.args[1] == 3
