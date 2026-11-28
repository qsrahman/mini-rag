"""Indexing pipeline: discover -> load -> chunk -> embed -> store (incrementally)."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from mini_rag.embeddings import Embedder
from mini_rag.ingestion.chunker import chunk_documents
from mini_rag.ingestion.loaders import (
    EmptyDocumentError,
    LoaderError,
    discover_files,
    document_id_for,
    load_file,
)
from mini_rag.models import Document, MiniRagError
from mini_rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Increase whenever loading or chunking changes the text or metadata stored for a file, so that
# `index` re-processes files it has already seen. A test pins the current output to this number.
INDEX_VERSION = 1


@dataclass
class IndexSummary:
    discovered: int = 0
    indexed: int = 0
    unchanged: int = 0
    chunks_created: int = 0
    pruned: int = 0  # documents removed because their file no longer exists
    failed: list[tuple[Path, str]] = field(default_factory=list)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)  # folders not listed


@dataclass(frozen=True)
class Scan:
    files: list[Path]
    unreadable: list[tuple[Path, str]]


def scan(directory: Path, root: Path, ignore: Sequence[str] = ()) -> Scan:
    """Find the files to index, noting subfolders that cannot be read."""
    unreadable: list[tuple[Path, str]] = []

    def note(folder: Path, error: OSError) -> None:
        unreadable.append((folder, str(error)))

    return Scan(discover_files(directory, root, ignore, note), unreadable)


class Indexer:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
        root: Path | None = None,
        ignore: Sequence[str] = (),
    ) -> None:
        """`root` is the project folder that document paths are relative to; without it, the
        directory being indexed is its own root."""
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._root = root
        self._ignore = ignore

    def index_directory(self, directory: Path, prune: bool = False) -> IndexSummary:
        directory = directory.resolve()
        root = (self._root or directory).resolve()
        if not directory.is_relative_to(root):
            raise MiniRagError(f"{directory} is outside the project {root}")
        found = scan(directory, root, self._ignore)
        files = found.files
        summary = IndexSummary(discovered=len(files), unreadable=found.unreadable)
        for folder, reason in found.unreadable:
            logger.warning("Could not read folder %s: %s", folder, reason)
        for path in files:
            try:
                created = self._index_file(path, root)
            except EmptyDocumentError as exc:
                # The content really is gone, so answering from the old version would be wrong.
                # A file that merely fails to parse keeps its old content (next branch).
                reason = str(exc)
                if self._store.delete_documents([self._document_id(path, root)]):
                    reason += " (removed previously indexed content)"
                logger.warning("Failed to index %s: %s", path, reason)
                summary.failed.append((path, reason))
                continue
            except (LoaderError, OSError) as exc:
                # Per-file problems (bad content, unreadable file) are reported, not fatal.
                # Ollama/store errors are not caught: they affect every file, so they abort the run.
                logger.warning("Failed to index %s: %s", path, exc)
                summary.failed.append((path, str(exc)))
                continue
            if created is None:
                summary.unchanged += 1
            else:
                summary.indexed += 1
                summary.chunks_created += created
        if prune:
            present = {self._document_id(f, root) for f in files}
            unseen = [PurePosixPath(folder.relative_to(root)) for folder, _ in found.unreadable]
            summary.pruned = self._prune(
                PurePosixPath(directory.relative_to(root)), present, unseen
            )
        return summary

    def _prune(
        self, directory: PurePosixPath, present: set[str], unseen: list[PurePosixPath]
    ) -> int:
        """Remove indexed documents under `directory` whose files are gone or now ignored.

        Files that exist are in `present`, so ones that failed to parse keep their old content.
        Nothing is removed under `unseen` folders: we could not tell what is still there.
        """
        gone = [
            document_id
            for document_id, source in self._store.indexed_documents().items()
            if document_id not in present
            and PurePosixPath(source).is_relative_to(directory)
            and not any(PurePosixPath(source).is_relative_to(folder) for folder in unseen)
        ]
        if gone and not present:
            # An empty or unmounted directory must not wipe the index.
            raise MiniRagError(
                f"Refusing to prune: no supported files found, "
                f"but {len(gone)} indexed document(s) live there."
            )
        self._store.delete_documents(gone)
        return len(gone)

    @staticmethod
    def _document_id(path: Path, root: Path) -> str:
        return document_id_for(path.relative_to(root).as_posix())

    def _fingerprint(self, data: bytes) -> str:
        # Chunking parameters and the index version are part of the fingerprint, so changing
        # either re-indexes. Version 1 adds nothing: indexes built before versions stay current.
        settings = f"|{self._chunk_size}|{self._chunk_overlap}"
        if INDEX_VERSION > 1:
            settings += f"|v{INDEX_VERSION}"
        digest = hashlib.sha256(data)
        digest.update(settings.encode())
        return digest.hexdigest()

    def _index_file(self, path: Path, root: Path) -> int | None:
        """Returns the number of chunks stored, or None if the file was already current."""
        document_id = self._document_id(path, root)
        data = path.read_bytes()
        content_hash = self._fingerprint(data)
        if self._store.is_current(document_id, content_hash):
            return None
        documents = [
            Document(d.text, {**d.metadata, "content_hash": content_hash})
            for d in load_file(path, root, data)
        ]
        chunks = chunk_documents(documents, self._chunk_size, self._chunk_overlap)
        # Embed everything before touching the store, so a failure leaves old content intact.
        embeddings = self._embedder.embed([c.text for c in chunks])
        self._store.replace_document(document_id, chunks, embeddings)
        return len(chunks)
