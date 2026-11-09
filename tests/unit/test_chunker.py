import itertools

import pytest

from mini_rag.ingestion.chunker import CHARS_PER_TOKEN, chunk_documents, split_text
from mini_rag.models import Document


def test_short_text_is_one_chunk():
    assert split_text("Just one short paragraph.", 100, 10) == ["Just one short paragraph."]


@pytest.mark.parametrize("text", ["", "   ", "\n\n\n"])
def test_empty_content_yields_no_chunks(text):
    assert split_text(text, 100, 10) == []


def test_paragraph_boundaries_are_preserved():
    paragraphs = [f"Paragraph {i} " + "word " * 30 for i in range(6)]
    chunks = split_text("\n\n".join(p.strip() for p in paragraphs), chunk_size=100, chunk_overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        for part in chunk.split("\n\n"):
            assert part in [p.strip() for p in paragraphs]


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(300))
    for chunk in split_text(text, chunk_size=50, chunk_overlap=10):
        assert len(chunk) <= 50 * CHARS_PER_TOKEN


def test_sentences_are_not_split_when_they_fit():
    text = " ".join(f"This is sentence {i}." for i in range(100))
    for chunk in split_text(text, chunk_size=40, chunk_overlap=0):
        assert chunk.endswith(".")
        assert chunk.startswith("This")


def test_overlap_repeats_trailing_text():
    text = "\n\n".join(f"Paragraph {i}. " + "filler " * 20 for i in range(8))
    chunks = split_text(text, chunk_size=100, chunk_overlap=50)
    assert len(chunks) > 1
    for previous, current in itertools.pairwise(chunks):
        last_paragraph = previous.split("\n\n")[-1]
        assert last_paragraph in current


def test_no_overlap_means_no_repeated_paragraphs():
    text = "\n\n".join(f"Paragraph {i}. " + "filler " * 20 for i in range(8))
    chunks = split_text(text, chunk_size=100, chunk_overlap=0)
    joined = "\n\n".join(chunks)
    assert all(joined.count(f"Paragraph {i}.") == 1 for i in range(8))


def test_oversized_word_run_is_hard_wrapped():
    chunks = split_text("x" * 5000, chunk_size=100, chunk_overlap=0)
    assert all(len(c) <= 400 for c in chunks)
    assert "".join(chunks) == "x" * 5000


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1), (10, 20)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", size, overlap)


def test_metadata_is_preserved_and_chunk_index_spans_pages():
    long_text = "\n\n".join("para " * 60 for _ in range(4))
    docs = [
        Document(long_text, {"source": "a.pdf", "document_id": "d1", "page": 1}),
        Document(long_text, {"source": "a.pdf", "document_id": "d1", "page": 2}),
        Document("other", {"source": "b.txt", "document_id": "d2"}),
    ]
    chunks = chunk_documents(docs, chunk_size=80, chunk_overlap=0)
    d1 = [c for c in chunks if c.metadata["document_id"] == "d1"]
    assert [c.metadata["chunk_index"] for c in d1] == list(range(len(d1)))
    assert {c.metadata["page"] for c in d1} == {1, 2}
    assert all(c.metadata["source"] == "a.pdf" for c in d1)
    [d2] = [c for c in chunks if c.metadata["document_id"] == "d2"]
    assert d2.metadata == {"source": "b.txt", "document_id": "d2", "chunk_index": 0}
    assert all(c.text.strip() for c in chunks)
