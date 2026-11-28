import os
from pathlib import Path

import pytest
from docx import Document as WordDocument

from mini_rag.ingestion.loaders import LoaderError, discover_files, load_file
from tests.conftest import make_docx, make_pdf


def test_txt_and_markdown(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.md").write_text("# Title\n\nbody")
    [a] = load_file(tmp_path / "a.txt", tmp_path)
    [b] = load_file(tmp_path / "b.md", tmp_path)
    assert a.text == "hello world"
    assert a.metadata["source"] == "a.txt"
    assert "page" not in a.metadata
    assert b.text.startswith("# Title")
    assert a.metadata["document_id"] != b.metadata["document_id"]


def test_pdf_pages_are_numbered_and_blank_pages_skipped(tmp_path):
    make_pdf(tmp_path / "doc.pdf", ["first page text", "", "third page text"])
    docs = load_file(tmp_path / "doc.pdf", tmp_path)
    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert "third" in docs[1].text


def test_pdf_without_text_is_an_error(tmp_path):
    make_pdf(tmp_path / "blank.pdf", ["", ""])
    with pytest.raises(LoaderError, match="no extractable text"):
        load_file(tmp_path / "blank.pdf")


def test_docx(tmp_path):
    make_docx(tmp_path / "d.docx", ["First paragraph.", "Second paragraph."])
    [doc] = load_file(tmp_path / "d.docx", tmp_path)
    assert "First paragraph." in doc.text and "Second paragraph." in doc.text
    assert doc.metadata["source"] == "d.docx"


def test_docx_preserves_paragraph_and_table_order(tmp_path):
    path = tmp_path / "ordered.docx"
    document = WordDocument()
    document.add_paragraph("Before table.")
    document.add_table(rows=1, cols=1).cell(0, 0).text = "Table content."
    document.add_paragraph("After table.")
    document.save(path)

    [loaded] = load_file(path, tmp_path)
    assert loaded.text == "Before table.\n\nTable content.\n\nAfter table."


def test_relative_source_for_nested_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "n.txt").write_text("nested")
    [doc] = load_file(tmp_path / "sub" / "n.txt", tmp_path)
    assert doc.metadata["source"] == "sub/n.txt"


@pytest.mark.parametrize("name", ["empty.txt", "ws.md"])
def test_empty_files_are_errors(tmp_path, name):
    (tmp_path / name).write_text("  \n\n ")
    with pytest.raises(LoaderError, match="no extractable text"):
        load_file(tmp_path / name)


@pytest.mark.parametrize("name", ["bad.pdf", "bad.docx"])
def test_corrupted_files_raise_loader_error(tmp_path, name):
    (tmp_path / name).write_bytes(b"this is not a real file")
    with pytest.raises(LoaderError, match=name):
        load_file(tmp_path / name)


def test_non_utf8_text_is_an_error(tmp_path):
    (tmp_path / "x.txt").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(LoaderError, match="UTF-8"):
        load_file(tmp_path / "x.txt")


def test_unsupported_format(tmp_path):
    (tmp_path / "x.exe").write_bytes(b"MZ")
    with pytest.raises(LoaderError, match="unsupported"):
        load_file(tmp_path / "x.exe")


def test_discover_is_recursive_and_filters(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / ".hidden").mkdir()
    for rel in ["a.txt", "sub/b.MD", "sub/c.png", ".hidden/d.txt", ".e.txt", "~$f.docx"]:
        (tmp_path / rel).write_text("x")
    found = [p.relative_to(tmp_path).as_posix() for p in discover_files(tmp_path)]
    assert found == ["a.txt", "sub/b.MD"]


def test_load_file_uses_supplied_bytes_instead_of_reading_the_file(tmp_path):
    [doc] = load_file(tmp_path / "never-created.txt", tmp_path, data=b"from memory")
    assert doc.text == "from memory"


def test_discover_does_not_descend_into_hidden_directories(tmp_path, monkeypatch):
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / ".git" / "objects" / "x.txt").write_text("x")
    (tmp_path / "keep.txt").write_text("x")
    visited = []
    real_walk = os.walk

    def spy(top, **kwargs):
        for directory, subdirs, names in real_walk(top, **kwargs):
            visited.append(Path(directory).name)
            yield directory, subdirs, names

    monkeypatch.setattr(os, "walk", spy)
    assert [p.name for p in discover_files(tmp_path)] == ["keep.txt"]
    assert ".git" not in visited and "objects" not in visited


def test_discover_surfaces_directory_scan_errors(tmp_path, monkeypatch):
    def fail_walk(top, onerror):
        onerror(PermissionError(13, "permission denied", str(tmp_path / "private")))

    monkeypatch.setattr(os, "walk", fail_walk)
    with pytest.raises(LoaderError, match=r"Could not scan directory.*private"):
        discover_files(tmp_path)


def test_ids_depend_only_on_the_path_inside_the_project(tmp_path):
    for name in ("one", "two"):
        (tmp_path / name / "sub").mkdir(parents=True)
        (tmp_path / name / "sub" / "n.txt").write_text("same")
    [a] = load_file(tmp_path / "one" / "sub" / "n.txt", tmp_path / "one")
    [b] = load_file(tmp_path / "two" / "sub" / "n.txt", tmp_path / "two")
    assert (
        a.metadata
        == b.metadata
        == {"source": "sub/n.txt", "document_id": a.metadata["document_id"]}
    )


@pytest.mark.parametrize(
    "patterns,expected",
    [
        ([], ["a.md", "build/b.md", "docs/build/c.md", "docs/d.txt"]),
        (["build"], ["a.md", "docs/d.txt"]),
        (["/build"], ["a.md", "docs/build/c.md", "docs/d.txt"]),
        (["*.txt"], ["a.md", "build/b.md", "docs/build/c.md"]),
        (["docs/build"], ["a.md", "build/b.md", "docs/d.txt"]),
    ],
)
def test_ignore_patterns(tmp_path, patterns, expected):
    for rel in ["a.md", "build/b.md", "docs/build/c.md", "docs/d.txt"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    found = discover_files(tmp_path, ignore=patterns)
    assert sorted(f.relative_to(tmp_path).as_posix() for f in found) == expected


def test_ignore_paths_are_relative_to_the_project_when_scanning_a_subfolder(tmp_path):
    (tmp_path / "docs" / "build").mkdir(parents=True)
    (tmp_path / "docs" / "build" / "c.md").write_text("x")
    (tmp_path / "docs" / "d.md").write_text("x")
    found = discover_files(tmp_path / "docs", tmp_path, ["docs/build"])
    assert [f.name for f in found] == ["d.md"]
