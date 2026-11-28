"""A long-running reader (chat) must follow changes made by `index` in another process."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from chromadb.errors import InternalError

from mini_rag.vector_store import IncompatibleEmbeddingModelError, VectorStore, VectorStoreError

ROOT = Path(__file__).resolve().parents[2]

# Runs in a separate interpreter: ChromaDB caches its vector index per process, so a second
# VectorStore in this process would not reproduce the problem.
_WRITER = """
import sys
from pathlib import Path
from mini_rag.models import Chunk
from mini_rag.vector_store import VectorStore

path, action = Path(sys.argv[1]), sys.argv[2]
model = "other-model" if action == "new-model" else "m"
store = VectorStore(path, "documents", model, rebuild=action in ("rebuild", "new-model"))
if action == "delete":
    store.delete_documents(["a"])
else:
    name, version = ("a", "h2") if action in ("replace", "rebuild") else ("b", "h1")
    meta = {"source": name, "document_id": name, "content_hash": version, "chunk_index": 0}
    store.replace_document(name, [Chunk(f"{action} text", meta)], [[0.0, 1.0, 0.0]])
"""


def other_process(path: Path, action: str) -> None:
    subprocess.run([sys.executable, "-c", _WRITER, str(path), action], check=True, cwd=ROOT)


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "db"
    subprocess.run(
        [sys.executable, "-c", _WRITER.replace('("b", "h1")', '("a", "h1")'), str(path), "seed"],
        check=True,
        cwd=ROOT,
    )
    return path


@pytest.fixture
def reader(path):
    store = VectorStore(path, "documents", "m")
    assert [r.chunk.text for r in store.query([0.0, 1.0, 0.0], 5)] == ["seed text"]  # load index
    return store


def dense(store):
    return sorted(r.chunk.text for r in store.query([0.0, 1.0, 0.0], 5))


def keyword(store):
    return sorted(r.chunk.text for r in store.keyword_query("text", 5))


@pytest.mark.parametrize(
    "action,expected",
    [
        ("add", ["add text", "seed text"]),
        ("replace", ["replace text"]),
        ("delete", []),
        ("rebuild", ["rebuild text"]),
    ],
)
def test_reader_follows_another_process(path, reader, action, expected):
    other_process(path, action)
    assert dense(reader) == expected
    assert keyword(reader) == expected
    assert reader.count() == len(expected)


def test_reader_keeps_following_repeated_changes(path, reader):
    for action, expected in [("add", 2), ("delete", 1), ("rebuild", 1), ("add", 2)]:
        other_process(path, action)
        assert len(dense(reader)) == expected


def test_count_works_first_after_a_rebuild(path, reader):
    other_process(path, "rebuild")
    assert reader.count() == 1


def test_rebuild_with_another_embedding_model_is_reported(path, reader):
    other_process(path, "new-model")
    with pytest.raises(IncompatibleEmbeddingModelError, match="other-model"):
        reader.query([0.0, 1.0, 0.0], 5)


def test_collection_missing_mid_rebuild_is_reported_and_recovers(path, reader):
    # The moment between a rebuild dropping the collection and creating it again.
    drop = (
        "import sys, chromadb; from mini_rag.keyword_index import KeywordIndex\n"
        "from mini_rag.vector_store import _keyword_table\n"
        "from pathlib import Path\n"
        "chromadb.PersistentClient(path=sys.argv[1]).delete_collection('documents')\n"
        "KeywordIndex(Path(sys.argv[1]) / 'keywords.sqlite3', _keyword_table('documents')).clear()"
    )
    subprocess.run([sys.executable, "-c", drop, str(path)], check=True, cwd=ROOT)
    with pytest.raises(VectorStoreError, match="being rebuilt"):
        reader.query([0.0, 1.0, 0.0], 5)

    other_process(path, "add")  # the reader must not have created a collection of its own
    assert dense(reader) == ["add text"]


def test_unchanged_index_is_not_reopened(reader):
    client = reader._client
    dense(reader)
    keyword(reader)
    assert reader._client is client


def flaky(store, failures, result=None):
    """Make the next `failures` Chroma queries misbehave like reads during indexing do."""
    # The unbound method: a bound one would keep calling the collection from before a reopen.
    real = type(store._collection).query
    calls = []

    def query(self, *args, **kwargs):
        calls.append(1)
        if len(calls) <= failures:
            if result is None:
                raise InternalError("Error creating hnsw segment reader: Nothing found on disk")
            return result
        return real(self, *args, **kwargs)

    return patch.object(type(store._collection), "query", query), calls


VANISHED = {"documents": [[None]], "metadatas": [[None]], "distances": [[0.1]]}


@pytest.mark.parametrize("result", [None, VANISHED], ids=["internal-error", "vanished-chunk"])
def test_reads_during_indexing_are_retried(reader, result):
    patched, calls = flaky(reader, 2, result)
    with patched:
        assert dense(reader) == ["seed text"]
    assert len(calls) == 3


def test_persistent_read_errors_become_a_clear_message(reader):
    patched, _ = flaky(reader, 100)
    with patched, pytest.raises(VectorStoreError, match="being updated"):
        reader.query([0.0, 1.0, 0.0], 5)


def test_chunks_that_stay_missing_are_dropped_not_crashed_on(reader):
    patched, _ = flaky(reader, 100, VANISHED)
    with patched:
        assert reader.query([0.0, 1.0, 0.0], 5) == []
