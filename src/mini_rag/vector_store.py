"""Persistent ChromaDB storage with safe per-document replacement."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import InvalidArgumentError, NotFoundError

from mini_rag.keyword_index import KeywordIndex
from mini_rag.models import Chunk, Metadata, MiniRagError, RetrievedChunk

logger = logging.getLogger(__name__)

_WRITE_BATCH = 1000


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
        self._embedding_model = embedding_model
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
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
        self._keywords = KeywordIndex(path / "keywords.sqlite3", _keyword_table(collection_name))
        if rebuild:
            self._keywords.clear()
        self._sync_keywords()

    def _sync_keywords(self) -> None:
        """Rebuild the keyword index from ChromaDB if the two disagree (first run after
        upgrade, or an interrupted write)."""
        stored = self._collection.get(include=["documents", "metadatas"])
        if set(stored["ids"]) == self._keywords.ids():
            return
        logger.info("Rebuilding keyword index from %d chunks", len(stored["ids"]))
        self._keywords.clear()
        self._keywords.add(
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
        """document_id -> absolute path ('' for documents indexed before paths were recorded)."""
        metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        return {str(m["document_id"]): str(m.get("path", "")) for m in metadatas}

    def delete_documents(self, document_ids: Iterable[str]) -> None:
        for document_id in document_ids:
            ids = self._collection.get(where={"document_id": document_id}, include=[])["ids"]
            if ids:
                self._collection.delete(ids=ids)
                self._keywords.delete(ids)

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
        self._keywords.delete([*ids, *stale])
        self._keywords.add(
            _keyword_row(cid, c.text, c.metadata) for cid, c in zip(ids, chunks, strict=True)
        )

    def keyword_query(
        self, text: str, top_k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """BM25 matches (no distance). Only `source` / `document_id` equality filters work."""
        filters = {k: v for k, v in (where or {}).items() if isinstance(v, str)}
        if len(filters) != len(where or {}):
            raise VectorStoreError("keyword search only supports string equality filters")
        try:
            ids = self._keywords.search(text, top_k, filters)
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
            RetrievedChunk(
                chunk=Chunk(
                    text=text,
                    metadata={k: v for k, v in meta.items() if isinstance(v, str | int)},
                ),
                distance=float(distance),
            )
            for text, meta, distance in zip(documents, metadatas, distances, strict=True)
        ]
