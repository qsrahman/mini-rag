"""Indexing pipeline: discover -> load -> chunk -> embed -> store (incrementally)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from mini_rag.embeddings import Embedder
from mini_rag.ingestion.chunker import chunk_documents
from mini_rag.ingestion.loaders import LoaderError, discover_files, document_id_for, load_file
from mini_rag.models import Document, MiniRagError
from mini_rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexSummary:
    discovered: int = 0
    indexed: int = 0
    unchanged: int = 0
    chunks_created: int = 0
    pruned: int = 0  # documents removed because their file no longer exists
    unprunable: int = 0  # documents indexed before paths were recorded; need --rebuild
    failed: list[tuple[Path, str]] = field(default_factory=list)


class Indexer:
    def __init__(
        self, embedder: Embedder, store: VectorStore, chunk_size: int, chunk_overlap: int
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def index_directory(self, root: Path, prune: bool = False) -> IndexSummary:
        files = discover_files(root)
        summary = IndexSummary(discovered=len(files))
        for path in files:
            try:
                created = self._index_file(path, root)
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
            self._prune(root, {document_id_for(f) for f in files}, summary)
        return summary

    def _prune(self, root: Path, present: set[str], summary: IndexSummary) -> None:
        """Remove indexed documents under `root` whose files are gone.

        Files that exist but failed to load are in `present`, so their old content is kept.
        """
        root = root.resolve()
        gone = []
        for document_id, path in self._store.indexed_documents().items():
            if document_id in present:
                continue
            if not path:
                summary.unprunable += 1
            elif Path(path).is_relative_to(root):
                gone.append(document_id)
        if gone and not present:
            # An empty or unmounted directory must not wipe the index.
            raise MiniRagError(
                f"Refusing to prune: no supported files found under {root}, "
                f"but {len(gone)} indexed document(s) live there."
            )
        self._store.delete_documents(gone)
        summary.pruned = len(gone)

    def _fingerprint(self, data: bytes) -> str:
        # Chunking parameters are part of the fingerprint so changing them re-indexes.
        digest = hashlib.sha256(data)
        digest.update(f"|{self._chunk_size}|{self._chunk_overlap}".encode())
        return digest.hexdigest()

    def _index_file(self, path: Path, root: Path) -> int | None:
        """Returns the number of chunks stored, or None if the file was already current."""
        document_id = document_id_for(path)
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
