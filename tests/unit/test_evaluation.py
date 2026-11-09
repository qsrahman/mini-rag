import math

import pytest

from mini_rag.evaluation import (
    EvalError,
    EvalQuestion,
    _clean_question,
    evaluate,
    format_report,
    generate_questions,
    load_questions,
    write_questions,
)
from mini_rag.models import Chunk, RetrievedChunk
from mini_rag.vector_store import chunk_id


def make_chunk(doc: str, index: int, text: str = "x") -> Chunk:
    return Chunk(text, {"document_id": doc, "content_hash": "h" * 12, "chunk_index": index})


def hit(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk, 0.1)


def question(chunk: Chunk, text: str = "q?") -> EvalQuestion:
    return EvalQuestion(text, chunk_id(chunk.metadata), str(chunk.metadata["document_id"]), "f.txt")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("What is X?", "What is X?"),
        ('  "What is X?"  ', "What is X?"),
        ("Question: What is X?\nExtra", "What is X?"),
        ("\n\n**Why?**", "Why?"),
        ("   \n", ""),
    ],
)
def test_clean_question(raw, expected):
    assert _clean_question(raw) == expected


def test_generate_questions_skips_empty_and_keeps_provenance():
    a, b = make_chunk("d1", 0, "alpha text"), make_chunk("d2", 3, "beta text")
    a = Chunk(a.text, {**a.metadata, "source": "a.pdf", "page": 4})
    answers = {"alpha text": "About alpha?", "beta text": ""}
    result = generate_questions(
        [(chunk_id(a.metadata), a), (chunk_id(b.metadata), b)], answers.__getitem__
    )
    assert result == [EvalQuestion("About alpha?", chunk_id(a.metadata), "d1", "a.pdf", 4)]


def test_question_file_roundtrip(tmp_path):
    qs = [EvalQuestion("a?", "id1", "d1", "x.md"), EvalQuestion("b?", "id2", "d2", "y.pdf", 2)]
    path = tmp_path / "sub" / "q.jsonl"
    write_questions(path, qs)
    assert load_questions(path) == qs


def test_load_questions_errors(tmp_path):
    with pytest.raises(EvalError, match="Cannot read"):
        load_questions(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    with pytest.raises(EvalError, match="not a valid"):
        load_questions(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(EvalError, match="no questions"):
        load_questions(empty)


def test_metrics_for_known_ranks():
    gold = [make_chunk("d1", i) for i in range(3)]
    other = make_chunk("d9", 0)
    qs = [question(g, f"q{i}") for i, g in enumerate(gold)]
    # q0: gold at rank 1; q1: gold at rank 3; q2: gold missing (but same doc at rank 2)
    results = {
        "q0": [hit(gold[0]), hit(other)],
        "q1": [hit(other), hit(other), hit(gold[1])],
        "q2": [hit(other), hit(make_chunk("d1", 99))],
    }
    report = evaluate(qs, lambda q, k: results[q][:k], {chunk_id(g.metadata) for g in gold}, k=5)

    assert report.questions == 3 and report.skipped == 0
    assert report.chunk_hit == {1: 1 / 3, 3: 2 / 3, 5: 2 / 3}
    assert report.doc_hit[3] == 1.0 and report.doc_hit[1] == 1 / 3
    assert report.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert report.ndcg == pytest.approx((1 + 1 / math.log2(4) + 0) / 3)
    assert report.misses == ["q2"]


def test_questions_with_vanished_chunks_are_skipped():
    live, gone = make_chunk("d1", 0), make_chunk("d1", 1)
    qs = [question(live, "a"), question(gone, "b")]
    report = evaluate(qs, lambda q, k: [hit(live)], {chunk_id(live.metadata)}, k=3)
    assert (report.questions, report.skipped) == (1, 1)
    assert report.chunk_hit[1] == 1.0
    assert "1 skipped" in format_report(report)


def test_all_questions_stale_is_an_error():
    qs = [question(make_chunk("d1", 0))]
    with pytest.raises(EvalError, match="Re-generate"):
        evaluate(qs, lambda q, k: [], set(), k=3)


def test_cutoffs_never_exceed_k():
    chunk = make_chunk("d1", 0)
    report = evaluate([question(chunk)], lambda q, k: [hit(chunk)], {chunk_id(chunk.metadata)}, k=2)
    assert list(report.chunk_hit) == [1, 2]
