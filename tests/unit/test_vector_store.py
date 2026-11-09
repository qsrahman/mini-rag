import pytest

from mini_rag.models import Chunk
from mini_rag.vector_store import IncompatibleEmbeddingModelError, VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "db", "docs", "model-a")


def chunks_for(document_id, texts, content_hash="h1", source="a.txt"):
    return [
        Chunk(
            t,
            {
                "source": source,
                "document_id": document_id,
                "content_hash": content_hash,
                "chunk_index": i,
                "page": 3,
            },
        )
        for i, t in enumerate(texts)
    ]


def vec(*xs):
    return list(xs)


def test_empty_collection(store):
    assert store.count() == 0
    assert store.query(vec(1.0, 0.0), top_k=5) == []
    assert not store.is_current("d1", "h1")


def test_add_and_query_preserves_text_and_metadata(store):
    chunks = chunks_for("d1", ["alpha", "beta"])
    store.replace_document("d1", chunks, [vec(1, 0), vec(0, 1)])
    [best] = store.query(vec(1, 0.1), top_k=1)
    assert best.chunk.text == "alpha"
    assert best.chunk.metadata == chunks[0].metadata
    assert best.distance < 0.1


def test_top_k_larger_than_collection(store):
    store.replace_document("d1", chunks_for("d1", ["a", "b"]), [vec(1, 0), vec(0, 1)])
    assert len(store.query(vec(1, 0), top_k=10)) == 2


def test_metadata_filter(store):
    store.replace_document("d1", chunks_for("d1", ["a"], source="one.txt"), [vec(1, 0)])
    store.replace_document("d2", chunks_for("d2", ["b"], source="two.txt"), [vec(1, 0)])
    results = store.query(vec(1, 0), top_k=5, where={"source": "two.txt"})
    assert [r.chunk.text for r in results] == ["b"]


def test_same_content_twice_does_not_duplicate(store):
    chunks = chunks_for("d1", ["a", "b"])
    for _ in range(2):
        store.replace_document("d1", chunks, [vec(1, 0), vec(0, 1)])
    assert store.count() == 2


def test_replacement_removes_old_chunks_only_for_that_document(store):
    store.replace_document("d1", chunks_for("d1", ["old1", "old2", "old3"]), [vec(1, 0)] * 3)
    store.replace_document("d2", chunks_for("d2", ["other"]), [vec(0, 1)])
    store.replace_document("d1", chunks_for("d1", ["new"], content_hash="h2"), [vec(1, 0)])
    texts = {r.chunk.text for r in store.query(vec(1, 0), top_k=10)}
    assert texts == {"new", "other"}
    assert store.is_current("d1", "h2")
    assert not store.is_current("d1", "h1")


def test_is_current_false_when_old_and_new_versions_coexist(store):
    # Simulates a crash between the upsert and the cleanup of a replacement.
    store.replace_document("d1", chunks_for("d1", ["old"]), [vec(1, 0)])
    store._collection.upsert(
        ids=["d1-h2-0"],
        documents=["new"],
        embeddings=[vec(1, 0)],
        metadatas=[dict(chunks_for("d1", ["new"], content_hash="h2")[0].metadata)],
    )
    assert not store.is_current("d1", "h2")


def test_length_mismatch_rejected(store):
    with pytest.raises(Exception, match="differ in length"):
        store.replace_document("d1", chunks_for("d1", ["a", "b"]), [vec(1, 0)])


def test_data_persists_across_instances(tmp_path):
    first = VectorStore(tmp_path / "db", "docs", "model-a")
    first.replace_document("d1", chunks_for("d1", ["a"]), [vec(1, 0)])
    second = VectorStore(tmp_path / "db", "docs", "model-a")
    assert second.count() == 1


def test_incompatible_embedding_model_is_rejected(tmp_path):
    VectorStore(tmp_path / "db", "docs", "model-a")
    with pytest.raises(IncompatibleEmbeddingModelError, match="--rebuild"):
        VectorStore(tmp_path / "db", "docs", "model-b")


def test_rebuild_clears_collection_and_allows_new_model(tmp_path):
    old = VectorStore(tmp_path / "db", "docs", "model-a")
    old.replace_document("d1", chunks_for("d1", ["a"]), [vec(1, 0)])
    rebuilt = VectorStore(tmp_path / "db", "docs", "model-b", rebuild=True)
    assert rebuilt.count() == 0
