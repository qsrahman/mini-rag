"""Persistent ChromaDB storage with safe per-document replacement."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import InternalError, InvalidArgumentError, NotFoundError

from mini_rag.keyword_index import KeywordIndex
from mini_rag.models import Chunk, Metadata, MiniRagError, RetrievedChunk

logger = logging.getLogger(__name__)

_WRITE_BATCH = 1000
_READ_ATTEMPTS = 5
_RETRY_DELAY = 0.1  # seconds, growing with each attempt


class VectorStoreError(MiniRagError):
    pass


class IncompatibleEmbeddingModelError(VectorStoreError):
    pass


def chunk_id(metadata: Metadata) -> str:
    """Stable and unique: changes whenever the file content or chunking changes."""
    return (
        f"{metadata['document_id']}-{str(metadata['content_hash'])[:12]}-{metadata['chunk_index']}"
    )


def _keyword_table(collection_name: str) -> str:
    return "kw_" + hashlib.sha1(collection_name.encode()).hexdigest()[:16]


def _keyword_row(cid: str, text: str, metadata: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return cid, str(metadata.get("source", "")), str(metadata.get("document_id", "")), text


class VectorStore:
    def __init__(
        self,
        path: Path,
        collection_name: str,
        embedding_model: str,
        rebuild: bool = False,
    ) -> None:
        self._path = path
        self._embedding_model = embedding_model
        self._collection_name = collection_name
        self._keywords = KeywordIndex(path / "keywords.sqlite3", _keyword_table(collection_name))
        # Read before loading anything, so a change made while we load is not missed.
        self._seen_version = self._keywords.data_version()
        self._client = self._new_client()
        if rebuild:
            self._delete_collection()
        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
            )
        except InvalidArgumentError as exc:
            raise VectorStoreError(f"Invalid collection name {collection_name!r}: {exc}") from exc
        self._check_embedding_model()
        if rebuild:
            self._keywords.clear()
        self._sync_keywords()

    def _new_client(self) -> ClientAPI:
        # Chroma's settings would otherwise be read from a `.env` in the current folder, letting a
        # downloaded folder reconfigure it, including which classes it imports. The Rust backend is
        # pinned because the Python one unpickles files found in the database directory.
        chroma_settings = ChromaSettings(  # type: ignore[call-arg]  # pydantic-settings arg
            _env_file=None,
            chroma_api_impl="chromadb.api.rust.RustBindingsAPI",
            anonymized_telemetry=False,
        )
        return chromadb.PersistentClient(path=str(self._path), settings=chroma_settings)

    def _refresh_if_changed(self, force: bool = False) -> None:
        """Reopen ChromaDB after another process (e.g. `index` while `chat` runs) changed it.

        ChromaDB caches its vector index per process: without this, searches miss new chunks, and
        fail outright after a rebuild. Every write ends with a keyword-index commit, which is what
        signals the change. Nothing is written here, so we never get in the indexer's way.
        """
        version = self._keywords.data_version()
        if version == self._seen_version and not force:
            return
        logger.info("Index changed by another process; reopening")
        # close() releases the cached index; a new client alone would reuse it.
        self._client.close()  # type: ignore[attr-defined]  # missing from the ClientAPI stub
        self._client = self._new_client()
        try:
            self._collection = self._client.get_collection(self._collection_name)
        except NotFoundError as exc:
            raise VectorStoreError(
                "The index is being rebuilt by another process. Try again in a moment."
            ) from exc
        self._check_embedding_model()
        self._seen_version = version  # only after success, so a failure is retried

    def _sync_keywords(self) -> None:
        """Rebuild the keyword index from ChromaDB if the two disagree (first run after
        upgrade, an interrupted write, or duplicate rows)."""
        # Ids only: this runs on every command, so chunk text is fetched only when needed.
        ids = self._collection.get(include=[])["ids"]
        if self._keywords.row_count() == len(ids) and self._keywords.ids() == set(ids):
            return
        stored = self._collection.get(include=["documents", "metadatas"])
        logger.info("Rebuilding keyword index from %d chunks", len(stored["ids"]))
        self._keywords.rebuild(
            _keyword_row(cid, text, meta)
            for cid, text, meta in zip(
                stored["ids"], stored["documents"] or [], stored["metadatas"] or [], strict=True
            )
        )

    def _delete_collection(self) -> None:
        with contextlib.suppress(NotFoundError):  # nothing to rebuild if it doesn't exist
            self._client.delete_collection(self._collection_name)

    def _check_embedding_model(self) -> None:
        stored = (self._collection.metadata or {}).get("embedding_model")
        if stored != self._embedding_model:
            raise IncompatibleEmbeddingModelError(
                f"Collection {self._collection_name!r} was built with embedding model "
                f"{stored!r}, but {self._embedding_model!r} is configured. Vectors from "
                "different models are not comparable. Re-run indexing with --rebuild, "
                "or switch back to the original model."
            )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def count(self) -> int:
        self._refresh_if_changed()
        return self._collection.count()

    def chunk_ids(self) -> set[str]:
        return set(self._collection.get(include=[])["ids"])

    def sample(self, n: int, seed: int = 0, min_chars: int = 200) -> list[tuple[str, Chunk]]:
        """Reproducible random sample of (id, chunk), skipping tiny chunks."""
        stored = self._collection.get(include=["documents", "metadatas"])
        pool = sorted(
            (cid, Chunk(text, {k: v for k, v in meta.items() if isinstance(v, str | int)}))
            for cid, text, meta in zip(
                stored["ids"], stored["documents"] or [], stored["metadatas"] or [], strict=True
            )
            if len(text) >= min_chars
        )
        return random.Random(seed).sample(pool, min(n, len(pool)))

    def indexed_documents(self) -> dict[str, str]:
        """document_id -> source path relative to the project."""
        metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        return {str(m["document_id"]): str(m["source"]) for m in metadatas}

    def delete_documents(self, document_ids: Iterable[str]) -> int:
        """Remove documents; returns the number of chunks removed."""
        removed = 0
        for document_id in document_ids:
            ids = self._collection.get(where={"document_id": document_id}, include=[])["ids"]
            if ids:
                self._collection.delete(ids=ids)
                self._keywords.delete(ids)
                removed += len(ids)
        return removed

    def is_current(self, document_id: str, content_hash: str) -> bool:
        """True if the document is indexed and every stored chunk matches `content_hash`."""
        stored = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
        metadatas = stored["metadatas"] or []
        return bool(metadatas) and all(m.get("content_hash") == content_hash for m in metadatas)

    def replace_document(
        self, document_id: str, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Add the new chunks first, then delete the document's older ones.

        Chunk ids embed the content hash, so new chunks never overwrite old ones.
        If this is interrupted after the upsert, the document has both versions
        (never zero), and the next indexing run sees a hash mismatch and finishes the cleanup.
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunks and embeddings differ in length")
        ids = [chunk_id(c.metadata) for c in chunks]
        for start in range(0, len(ids), _WRITE_BATCH):
            end = start + _WRITE_BATCH
            self._collection.upsert(
                ids=ids[start:end],
                documents=[c.text for c in chunks[start:end]],
                embeddings=list(embeddings[start:end]),
                metadatas=[dict(c.metadata) for c in chunks[start:end]],
            )
        existing = self._collection.get(where={"document_id": document_id}, include=[])["ids"]
        stale = sorted(set(existing) - set(ids))
        if stale:
            self._collection.delete(ids=stale)
        self._keywords.replace(
            [*ids, *stale],
            (_keyword_row(cid, c.text, c.metadata) for cid, c in zip(ids, chunks, strict=True)),
        )

    def keyword_query(
        self, text: str, top_k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """BM25 matches (no distance). Only `source` / `document_id` equality filters work."""
        self._refresh_if_changed()
        filters = {k: v for k, v in (where or {}).items() if isinstance(v, str)}
        if len(filters) != len(where or {}):
            raise VectorStoreError("keyword search only supports string equality filters")
        try:
            # Chroma rejects repeated ids, so never let a duplicate row break a query.
            ids = list(dict.fromkeys(self._keywords.search(text, top_k, filters)))
        except ValueError as exc:
            raise VectorStoreError(str(exc)) from exc
        if not ids:
            return []
        stored = self._collection.get(ids=ids, include=["documents", "metadatas"])
        by_id = {
            cid: Chunk(doc, {k: v for k, v in meta.items() if isinstance(v, str | int)})
            for cid, doc, meta in zip(
                stored["ids"], stored["documents"] or [], stored["metadatas"] or [], strict=True
            )
        }
        return [RetrievedChunk(by_id[cid]) for cid in ids if cid in by_id]

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Dense search. Retries while another process is part-way through writing, when
        ChromaDB may fail or return chunks that have just been deleted."""
        failure: Exception | None = None
        for attempt in range(_READ_ATTEMPTS):
            if attempt:
                time.sleep(_RETRY_DELAY * attempt)
            self._refresh_if_changed(force=attempt > 0)
            try:
                found = self._dense_search(embedding, top_k, where)
            except (InternalError, NotFoundError) as exc:
                failure = exc
                continue
            complete = [r for r in found if r is not None]
            if len(complete) == len(found) or attempt == _READ_ATTEMPTS - 1:
                return complete
        raise VectorStoreError(
            "The index is being updated by another process and could not be read. "
            f"Try again in a moment. ({failure})"
        ) from failure

    def _dense_search(
        self, embedding: Sequence[float], top_k: int, where: dict[str, Any] | None
    ) -> list[RetrievedChunk | None]:
        """One attempt; None stands for a match whose chunk no longer exists."""
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            None
            if text is None or meta is None
            else RetrievedChunk(
                chunk=Chunk(
                    text=text,
                    metadata={k: v for k, v in meta.items() if isinstance(v, str | int)},
                ),
                distance=float(distance),
            )
            for text, meta, distance in zip(documents, metadatas, distances, strict=True)
        ]
