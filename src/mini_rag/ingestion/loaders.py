"""Document loaders: file -> list of Documents with source metadata."""

from __future__ import annotations

import hashlib
import io
import logging
import os
from collections.abc import Callable
from pathlib import Path

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


def document_id_for(path: Path) -> str:
    """Stable id derived from the absolute path."""
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]


def discover_files(root: Path) -> list[Path]:
    """Recursively find supported files, skipping hidden entries and Word lock files."""
    found: list[Path] = []

    def onerror(error: OSError) -> None:
        location = error.filename or str(root)
        raise LoaderError(f"Could not scan directory {location}: {error}") from error

    for directory, subdirs, names in os.walk(root, onerror=onerror):
        subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))  # prune, don't descend
        found.extend(
            Path(directory, name)
            for name in sorted(names)
            if not name.startswith((".", "~$"))
            and Path(name).suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return found


def load_file(path: Path, root: Path | None = None, data: bytes | None = None) -> list[Document]:
    """Extract text from one file (`data` = its bytes, if the caller already read them).

    Raises LoaderError if nothing usable comes out.
    """
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise LoaderError(f"{path.name}: unsupported file format {path.suffix!r}")
    try:
        pages = reader(path.read_bytes() if data is None else data)
    except Exception as exc:  # parser libraries raise many unrelated exception types
        raise LoaderError(f"{path.name}: could not read file ({exc})") from exc

    source = path.relative_to(root).as_posix() if root else path.name
    document_id = document_id_for(path)
    documents = []
    for page, text in pages:
        if not text.strip():
            if page is not None:
                logger.warning("%s: page %d has no extractable text (scanned?)", path.name, page)
            continue
        # `path` lets `index --prune` tell which indexed documents live under a given directory.
        metadata: Metadata = {
            "source": source,
            "document_id": document_id,
            "path": str(path.resolve()),
        }
        if page is not None:
            metadata["page"] = page
        documents.append(Document(text=text, metadata=metadata))
    if not documents:
        raise LoaderError(f"{path.name}: no extractable text")
    return documents
