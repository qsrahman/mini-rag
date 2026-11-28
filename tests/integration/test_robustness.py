"""Unreadable folders, safe rebuilds, and clean errors instead of tracebacks."""

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from mini_rag.cli import main
from mini_rag.embeddings import OllamaEmbedder
from mini_rag.ollama_utils import OllamaError
from mini_rag.project import init_project
from mini_rag.vector_store import VectorStore
from tests.conftest import FakeEmbedder


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "proj"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("alpha content here")
    (root / "sub" / "b.txt").write_text("beta content here")
    init_project(root)
    return root


@pytest.fixture
def args(root, monkeypatch):
    monkeypatch.setattr("mini_rag.cli.OllamaEmbedder", lambda model, host: FakeEmbedder(model))
    return ["--project", str(root), "--embedding-model", "fake-embed"]


@pytest.fixture
def indexed(args, capsys):
    assert main(["index", *args]) == 0
    capsys.readouterr()
    return args


@pytest.fixture
def locked(root):
    """Make `sub` unreadable for the test."""
    folder = root / "sub"
    folder.chmod(0)
    yield folder
    folder.chmod(0o755)


def sources(root: Path) -> list[str]:
    store = VectorStore(root / ".mini-rag" / "chroma", "documents", "fake-embed")
    return sorted(store.indexed_documents().values())


def error_of(capsys, argv) -> str:
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    return err


# --- unreadable folders ---


def test_unreadable_folder_does_not_stop_indexing(root, indexed, locked, capsys):
    (root / "new.txt").write_text("a brand new file")

    assert main(["index", *indexed]) == 1

    out = capsys.readouterr()
    assert "Files indexed: 1" in out.out and "Folders unreadable: 1" in out.out
    assert "sub" in out.err
    locked.chmod(0o755)
    assert sources(root) == ["a.txt", "new.txt", "sub/b.txt"]


def test_prune_keeps_documents_under_an_unreadable_folder(root, indexed, locked, capsys):
    (root / "a.txt").unlink()
    (root / "new.txt").write_text("a brand new file")

    main(["index", "--prune", *indexed])

    assert "Documents pruned: 1" in capsys.readouterr().out  # a.txt only
    locked.chmod(0o755)
    assert sources(root) == ["new.txt", "sub/b.txt"]


def test_unreadable_target_folder_is_an_error(indexed, locked, capsys):
    assert "Could not scan" in error_of(capsys, ["index", str(locked), *indexed])


# --- rebuild never destroys the index before it knows it can refill it ---


def test_rebuild_is_refused_when_a_folder_is_unreadable(root, indexed, locked, capsys):
    err = error_of(capsys, ["index", "--rebuild", *indexed])
    assert "Not rebuilding" in err and "sub" in err
    locked.chmod(0o755)
    assert sources(root) == ["a.txt", "sub/b.txt"]


def test_ignoring_the_unreadable_folder_allows_a_rebuild(root, indexed, locked, capsys):
    (root / ".mini-rag" / "ignore").write_text("sub\n")
    assert main(["index", "--rebuild", *indexed]) == 0
    assert sources(root) == ["a.txt"]


def test_rebuild_is_refused_when_there_are_no_files(root, indexed, capsys):
    (root / "a.txt").unlink()
    (root / "sub" / "b.txt").unlink()
    assert "Not rebuilding" in error_of(capsys, ["index", "--rebuild", *indexed])
    assert sources(root) == ["a.txt", "sub/b.txt"]


def test_rebuild_with_a_broken_ignore_file_keeps_the_index(root, indexed, capsys):
    (root / ".mini-rag" / "ignore").write_bytes(b"\xff\xfe")
    assert "ignore" in error_of(capsys, ["index", "--rebuild", *indexed])
    assert sources(root) == ["a.txt", "sub/b.txt"]


# --- clean errors ---


@pytest.mark.parametrize("name", ["config.env", "ignore"])
@pytest.mark.parametrize("damage", ["not-utf8", "no-permission"])
def test_unreadable_project_files(root, indexed, capsys, name, damage):
    path = root / ".mini-rag" / name
    if damage == "not-utf8":
        path.write_bytes(b"\xff\xfe")
    else:
        path.chmod(0)
    try:
        assert name in error_of(capsys, ["index", *indexed])
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize("count", ["-1", "0"])
def test_generate_needs_a_positive_count(indexed, tmp_path, capsys, count):
    err = error_of(capsys, ["eval", str(tmp_path / "q.jsonl"), "--generate", count, *indexed])
    assert "--generate must be positive" in err


def test_init_in_a_read_only_folder(tmp_path, capsys):
    folder = tmp_path / "ro"
    folder.mkdir()
    folder.chmod(0o500)
    try:
        assert "Permission denied" in error_of(capsys, ["init", str(folder)])
    finally:
        folder.chmod(0o755)


@pytest.mark.parametrize("host", ["http://[bad", "localhost:abc", "ftp://localhost"])
def test_malformed_ollama_host(root, capsys, host):
    argv = ["ask", "q", "--project", str(root), "--ollama-host", host]
    main(["status", "--project", str(root)])
    store = VectorStore(root / ".mini-rag" / "chroma", "documents", "nomic-embed-text-v2-moe")
    meta = {"source": "a", "document_id": "a", "content_hash": "h", "chunk_index": 0}
    from mini_rag.models import Chunk

    store.replace_document("a", [Chunk("text", meta)], [[1.0, 0.0]])
    capsys.readouterr()
    assert "Ollama" in error_of(capsys, argv)


@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("dropped")]
)
def test_transport_failures_are_ollama_errors(failure):
    client = MagicMock()
    client.embed.side_effect = failure
    with pytest.raises(OllamaError, match="Request to Ollama"):
        OllamaEmbedder("m", client=client).embed(["a"])


def test_damaged_keyword_index_says_how_to_recover(root, indexed, capsys):
    database = root / ".mini-rag" / "chroma" / "keywords.sqlite3"
    database.write_bytes(b"corrupt" * 200)
    err = error_of(capsys, ["status", *indexed])
    assert "keywords.sqlite3" in err and "delete" in err.lower()

    database.unlink()  # following the advice works
    assert main(["status", *indexed]) == 0
    assert "Indexed chunks: 2" in capsys.readouterr().out


def test_damaged_vector_database_is_reported(root, args):
    # In a fresh process: an already open Chroma would not re-read the file.
    import subprocess
    import sys

    database = root / ".mini-rag" / "chroma"
    database.mkdir()
    (database / "chroma.sqlite3").write_bytes(b"corrupt" * 200)
    result = subprocess.run(
        [sys.executable, "-m", "mini_rag", "status", "--project", str(root)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr and "error:" in result.stderr
