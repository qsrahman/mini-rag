from unittest.mock import MagicMock

import pytest

from mini_rag.models import Chunk, RetrievedChunk
from mini_rag.rag import (
    NO_CONTENT_MESSAGE,
    RagService,
    build_context,
    build_sources,
    find_unknown_citations,
)
from mini_rag.retriever import Retriever


def retrieved(text, source, page=None, distance=0.1):
    metadata = {"source": source}
    if page is not None:
        metadata["page"] = page
    return RetrievedChunk(Chunk(text, metadata), distance)


# --- Retriever ---


def test_retriever_embeds_question_and_queries_store(embedder):
    store = MagicMock()
    store.query.return_value = ["result"]
    result = Retriever(embedder, store, top_k=5).retrieve("what is x?")
    assert result == ["result"]
    assert embedder.calls == [["what is x?"]]
    args = store.query.call_args.args
    assert args[1] == 5 and args[2] is None


def test_retriever_top_k_and_filter_override(embedder):
    store = MagicMock()
    Retriever(embedder, store, top_k=5).retrieve("q", top_k=2, where={"source": "a"})
    assert store.query.call_args.args[1:] == (2, {"source": "a"})


@pytest.mark.parametrize("top_k", [0, -1])
def test_retriever_rejects_non_positive_top_k_override(embedder, top_k):
    store = MagicMock()
    with pytest.raises(ValueError, match="top_k must be positive"):
        Retriever(embedder, store).retrieve("q", top_k=top_k)
    assert embedder.calls == []
    store.query.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1])
def test_retriever_rejects_non_positive_default_top_k(embedder, top_k):
    with pytest.raises(ValueError, match="top_k must be positive"):
        Retriever(embedder, MagicMock(), top_k=top_k)


def test_retriever_empty_results(embedder):
    store = MagicMock()
    store.query.return_value = []
    assert Retriever(embedder, store).retrieve("q") == []


# --- RAG ---


def test_sources_and_context_are_labelled():
    items = [retrieved("Text one", "ml.pdf", page=12), retrieved("Text two", "intro.docx")]
    sources = build_sources(items)
    assert [(s.id, s.name, s.page) for s in sources] == [
        ("S1", "ml.pdf", 12),
        ("S2", "intro.docx", None),
    ]
    context = build_context(items, sources)
    assert '<document id="S1">\n<source>ml.pdf, page 12</source>' in context
    assert "<document_content>\nText one\n</document_content>" in context
    assert '<document id="S2">\n<source>intro.docx</source>' in context


def test_ask_passes_context_to_llm_and_returns_retrieved_sources():
    retriever, generator = MagicMock(), MagicMock()
    retriever.retrieve.return_value = [retrieved("Supervised learning uses labels.", "ml.pdf", 12)]
    generator.generate.return_value = "It uses labelled data [S1]."
    answer = RagService(retriever, generator).ask(
        "What is supervised learning?", where={"source": "ml.pdf"}
    )

    retriever.retrieve.assert_called_once_with(
        "What is supervised learning?", where={"source": "ml.pdf"}
    )
    question, context = generator.generate.call_args.args
    assert question == "What is supervised learning?"
    assert "Supervised learning uses labels." in context and 'id="S1"' in context
    assert answer.text == "It uses labelled data [S1]."
    assert [s.id for s in answer.sources] == ["S1"]
    assert answer.unknown_citations == []


def test_fabricated_citations_are_flagged_not_trusted():
    retriever, generator = MagicMock(), MagicMock()
    retriever.retrieve.return_value = [retrieved("t", "a.txt")]
    generator.generate.return_value = "Claim [S1] and another [S4, S1]."
    answer = RagService(retriever, generator).ask("q")
    assert [s.id for s in answer.sources] == ["S1"]
    assert answer.unknown_citations == ["S4"]


def test_no_retrieved_chunks_skips_llm():
    retriever, generator = MagicMock(), MagicMock()
    retriever.retrieve.return_value = []
    answer = RagService(retriever, generator).ask("anything")
    generator.generate.assert_not_called()
    assert answer.text == NO_CONTENT_MESSAGE
    assert answer.sources == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("see [S1] and [S2]", []),
        ("see [S3]", ["S3"]),
        ("[S1, S9] [S9]", ["S9"]),
        ("no brackets S7", []),
        ("[not a citation]", []),
    ],
)
def test_find_unknown_citations(text, expected):
    sources = build_sources([retrieved("a", "a"), retrieved("b", "b")])
    assert find_unknown_citations(text, sources) == expected
