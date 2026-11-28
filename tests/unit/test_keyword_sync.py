"""The keyword index must stay a faithful, duplicate-free copy of ChromaDB."""

from unittest.mock import patch

import pytest

from mini_rag.models import Chunk
from mini_rag.vector_store import VectorStore


def chunks(document: str, n: int = 3, version: str = "h1") -> list[Chunk]:
    return [
        Chunk(
            f"beta item {i}",
            {
                "source": document,
                "document_id": document,
                "content_hash": version,
                "chunk_index": i,
            },
        )
        for i in range(n)
    ]


def vectors(n: int = 3) -> list[list[float]]:
    return [[1.0, float(i)] for i in range(n)]


def rows(store: VectorStore) -> list[str]:
    table = store._keywords._table
    return [r[0] for r in store._keywords._db.execute(f"SELECT chunk_id FROM {table}")]


@pytest.fixture
def path(tmp_path):
    return tmp_path / "db"


@pytest.mark.parametrize("method", ["delete", "add", "replace"])
def test_another_open_during_a_write_does_not_duplicate_rows(path, method):
    writer = VectorStore(path, "documents", "m")
    real = getattr(writer._keywords, method, None)
    if real is None:
        pytest.skip(f"KeywordIndex.{method} does not exist")

    def open_elsewhere_first(*args):
        VectorStore(path, "documents", "m")  # e.g. `ask` starting while `index` runs
        return real(*args)

    with patch.object(writer._keywords, method, open_elsewhere_first):
        writer.replace_document("d", chunks("d"), vectors())

    reader = VectorStore(path, "documents", "m")
    assert sorted(rows(reader)) == sorted(reader.chunk_ids())
    assert len(reader.keyword_query("beta", 10)) == 3


def test_existing_duplicate_rows_are_repaired_on_open(path):
    store = VectorStore(path, "documents", "m")
    store.replace_document("d", chunks("d"), vectors())
    store._keywords.add([(rows(store)[0], "d", "d", "beta item 0")])
    assert len(rows(store)) == 4

    assert len(rows(VectorStore(path, "documents", "m"))) == 3


def test_duplicate_rows_never_break_a_query(path):
    store = VectorStore(path, "documents", "m")
    store.replace_document("d", chunks("d"), vectors())
    store._keywords.add([(rows(store)[0], "d", "d", "beta item 0")])
    assert len(store.keyword_query("beta", 10)) == 3


def test_a_failed_keyword_write_leaves_the_previous_rows(path):
    store = VectorStore(path, "documents", "m")
    store.replace_document("d", chunks("d"), vectors())
    before = sorted(rows(store))

    with pytest.raises(Exception, match="boom"):

        def exploding():
            yield ("new-id", "d", "d", "text")
            raise RuntimeError("boom")

        store._keywords.replace(before, exploding())

    assert sorted(rows(store)) == before


def test_opening_an_in_step_index_does_not_load_chunk_text(path):
    VectorStore(path, "documents", "m").replace_document("d", chunks("d"), vectors())
    from chromadb.api.models.Collection import Collection

    real = Collection.get
    requested = []

    def spy(self, *args, **kwargs):
        requested.append(kwargs.get("include"))
        return real(self, *args, **kwargs)

    with patch.object(Collection, "get", spy):
        VectorStore(path, "documents", "m")

    assert requested == [[]]
