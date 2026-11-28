"""Changes to loading or chunking must reach files that are already indexed."""

import hashlib

import docx
import pytest

from mini_rag import indexer as indexer_module
from mini_rag.indexer import Indexer
from mini_rag.ingestion.chunker import chunk_documents
from mini_rag.ingestion.loaders import load_file
from mini_rag.vector_store import VectorStore
from tests.conftest import make_pdf


@pytest.fixture
def indexed(tmp_path, embedder):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("alpha content")
    (root / "b.md").write_text("beta content")
    store = VectorStore(tmp_path / "db", "docs", embedder.model)
    indexer = Indexer(embedder, store, 200, 20)
    indexer.index_directory(root)
    return root, store, indexer


def test_new_index_version_reindexes_unchanged_files_once(indexed, monkeypatch):
    root, store, indexer = indexed
    old_ids = store.chunk_ids()
    monkeypatch.setattr(indexer_module, "INDEX_VERSION", indexer_module.INDEX_VERSION + 1)

    summary = indexer.index_directory(root)

    assert (summary.indexed, summary.unchanged) == (2, 0)
    assert store.count() == 2 and not store.chunk_ids() & old_ids  # replaced, not duplicated
    assert indexer.index_directory(root).unchanged == 2


def sample_output(tmp_path) -> str:
    """Digest of everything loading and chunking produce for a fixed set of documents."""
    prose = "\n\n".join(f"Paragraph {i}. " + "It has a sentence. " * 8 for i in range(6))
    (tmp_path / "prose.txt").write_text(prose)
    (tmp_path / "list.md").write_text("\n".join(f"- option_{i} enables {i}" for i in range(40)))
    (tmp_path / "bom.txt").write_bytes(b"\xef\xbb\xbfbyte order mark")
    make_pdf(tmp_path / "pages.pdf", ["First page text.", "", "Third page text."])
    document = docx.Document()
    document.add_paragraph("Before.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "left", "right"
    document.add_paragraph("After.")
    document.save(str(tmp_path / "table.docx"))

    produced = []
    for path in sorted(tmp_path.iterdir()):
        for chunk in chunk_documents(load_file(path, tmp_path), 60, 15):
            produced.append((chunk.text, sorted(chunk.metadata.items())))
    return hashlib.sha256(repr(produced).encode()).hexdigest()[:16]


def test_loader_and_chunker_output_is_pinned_to_the_index_version(tmp_path):
    expected = (2, "68b990edf997cc48")
    assert (indexer_module.INDEX_VERSION, sample_output(tmp_path)) == expected, (
        "Loading or chunking output changed. Files already indexed will keep the old output "
        "unless INDEX_VERSION in indexer.py is increased; then update `expected` here."
    )
