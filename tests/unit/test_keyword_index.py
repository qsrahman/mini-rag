import pytest

from mini_rag.keyword_index import KeywordIndex


@pytest.fixture
def index(tmp_path):
    kw = KeywordIndex(tmp_path / "kw.sqlite3", "kw_test")
    kw.add(
        [
            ("c1", "a.md", "d1", "Error code E1234 means the cache is full."),
            ("c2", "a.md", "d1", "Configure the module by editing the settings file."),
            ("c3", "b.md", "d2", "Indexing documents creates searchable chunks."),
        ]
    )
    return kw


def test_exact_identifier_is_found(index):
    assert index.search("what does E1234 mean?", 5)[0] == "c1"


def test_stemming_matches_word_forms(index):
    assert index.search("how do I index a document", 5)[0] == "c3"


def test_filters_restrict_results(index):
    assert index.search("module settings chunks", 5, {"source": "b.md"}) == ["c3"]
    assert index.search("module settings chunks", 5, {"document_id": "d1"}) == ["c2"]


def test_unknown_filter_column_is_rejected(index):
    with pytest.raises(ValueError, match="cannot filter"):
        index.search("x", 5, {"text": "y"})


@pytest.mark.parametrize("query", ["", "   ", "?!", '"quoted" AND NOT (x) OR *'])
def test_arbitrary_text_never_breaks_the_fts_query(index, query):
    assert isinstance(index.search(query, 5), list)


def test_limit_delete_and_clear(index):
    assert len(index.search("the module chunks error", 2)) == 2
    index.delete(["c1"])
    assert index.ids() == {"c2", "c3"}
    index.clear()
    assert index.ids() == set()
