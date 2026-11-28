"""Document loaders: file -> list of Documents with source metadata."""

from __future__ import annotations

import hashlib
import io
import logging
import os
from collections.abc import Callable, Sequence
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

import docx
import pymupdf
from docx.table import Table
from docx.text.paragraph import Paragraph

from mini_rag.models import Document, Metadata, MiniRagError

logger = logging.getLogger(__name__)

# Each reader returns (page_number_or_None, text) pairs.
_Pages = list[tuple[int | None, str]]


class LoaderError(MiniRagError):
    pass


class EmptyDocumentError(LoaderError):
    """The file was read successfully and holds no text (unlike one that could not be read)."""


def _read_text(data: bytes) -> _Pages:
    try:
        return [(None, data.decode("utf-8-sig"))]
    except UnicodeDecodeError as exc:
        raise LoaderError("not valid UTF-8 text") from exc


def _read_pdf(data: bytes) -> _Pages:
    with pymupdf.open(stream=data, filetype="pdf") as pdf:  # type: ignore[no-untyped-call]
        return [(number, str(page.get_text())) for number, page in enumerate(pdf, start=1)]


def _read_docx(data: bytes) -> _Pages:
    document = docx.Document(io.BytesIO(data))
    lines = []
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            lines.append(block.text)
        else:
            table: Table = block
            lines.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
    return [(None, "\n\n".join(lines))]


# Readers parse in-memory bytes so the indexer can hash and parse a single read of the file.
_READERS: dict[str, Callable[[bytes], _Pages]] = {
    ".txt": _read_text,
    ".md": _read_text,
    ".markdown": _read_text,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}
SUPPORTED_EXTENSIONS = frozenset(_READERS)


def document_id_for(source: str) -> str:
    """Stable id from the path relative to the project, so a moved project keeps its ids."""
    return hashlib.sha1(source.encode()).hexdigest()[:16]


def _is_ignored(relative: PurePosixPath, patterns: Sequence[str]) -> bool:
    return any(
        fnmatchcase(str(relative) if "/" in pattern else relative.name, pattern.lstrip("/"))
        for pattern in patterns
    )


def discover_files(
    directory: Path,
    root: Path | None = None,
    ignore: Sequence[str] = (),
    on_unreadable: Callable[[Path, OSError], None] | None = None,
) -> list[Path]:
    """Recursively find supported files, skipping hidden entries, Word lock files and anything
    matching an `ignore` pattern (paths in patterns are relative to `root`).

    A subfolder that cannot be listed is passed to `on_unreadable`; without that callback, or
    when `directory` itself cannot be listed, LoaderError is raised.
    """
    root = root or directory
    found: list[Path] = []

    def onerror(error: OSError) -> None:
        location = error.filename or str(directory)
        if on_unreadable and Path(location) != directory:
            on_unreadable(Path(location), error)
            return
        raise LoaderError(f"Could not scan directory {location}: {error}") from error

    def wanted(parent: str, name: str) -> bool:
        relative = PurePosixPath(Path(parent, name).relative_to(root).as_posix())
        return not name.startswith((".", "~$")) and not _is_ignored(relative, ignore)

    for parent, subdirs, names in os.walk(directory, onerror=onerror):
        subdirs[:] = sorted(d for d in subdirs if wanted(parent, d))  # prune, don't descend
        found.extend(
            Path(parent, name)
            for name in sorted(names)
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS and wanted(parent, name)
        )
    return found


def load_file(path: Path, root: Path | None = None, data: bytes | None = None) -> list[Document]:
    """Extract text from one file (`data` = its bytes, if the caller already read them).

    Raises LoaderError if the file cannot be read, EmptyDocumentError if it has no text.
    """
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise LoaderError(f"{path.name}: unsupported file format {path.suffix!r}")
    try:
        pages = reader(path.read_bytes() if data is None else data)
    except Exception as exc:  # parser libraries raise many unrelated exception types
        raise LoaderError(f"{path.name}: could not read file ({exc})") from exc

    source = path.relative_to(root).as_posix() if root else path.name
    document_id = document_id_for(source)
    documents = []
    for page, text in pages:
        if not text.strip():
            if page is not None:
                logger.warning("%s: page %d has no extractable text (scanned?)", path.name, page)
            continue
        metadata: Metadata = {"source": source, "document_id": document_id}
        if page is not None:
            metadata["page"] = page
        documents.append(Document(text=text, metadata=metadata))
    if not documents:
        raise EmptyDocumentError(f"{path.name}: no extractable text")
    return documents
