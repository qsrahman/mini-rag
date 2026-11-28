"""`status` reports on a project without creating or changing anything."""

import pytest

from mini_rag.cli import main
from mini_rag.project import init_project
from tests.conftest import FakeEmbedder


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr("mini_rag.cli.OllamaEmbedder", lambda model, host: FakeEmbedder(model))
    (tmp_path / "a.txt").write_text("alpha content")
    init_project(tmp_path)
    return tmp_path


def status(root, capsys, *extra) -> str:
    assert main(["status", "--project", str(root), *extra]) == 0
    return capsys.readouterr().out


def snapshot(root):
    data = root / ".mini-rag"
    return sorted(str(p.relative_to(data)) for p in data.rglob("*"))


def test_status_before_indexing_creates_nothing(root, capsys):
    before = snapshot(root)
    assert "Indexed chunks: 0" in status(root, capsys)
    assert snapshot(root) == before


def test_status_counts_and_warns_about_another_embedding_model(root, capsys):
    main(["index", "--project", str(root), "--embedding-model", "fake-embed"])
    capsys.readouterr()

    same = status(root, capsys, "--embedding-model", "fake-embed")
    assert "Indexed chunks: 1" in same and "Warning" not in same

    other = status(root, capsys, "--embedding-model", "something-else")
    assert "Indexed chunks: 1" in other
    assert "'fake-embed'" in other and "--rebuild" in other


def test_status_for_a_missing_collection(root, capsys):
    main(["index", "--project", str(root), "--embedding-model", "fake-embed"])
    capsys.readouterr()
    assert "Indexed chunks: 0" in status(root, capsys, "--collection", "elsewhere")
    assert "Indexed chunks: 1" in status(root, capsys, "--embedding-model", "fake-embed")


def test_status_does_not_repair_or_touch_the_keyword_index(root, capsys):
    main(["index", "--project", str(root), "--embedding-model", "fake-embed"])
    capsys.readouterr()
    keywords = root / ".mini-rag" / "chroma" / "keywords.sqlite3"
    keywords.unlink()
    status(root, capsys)
    assert not keywords.exists()
