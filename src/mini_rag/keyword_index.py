"""BM25 keyword search over chunk text using SQLite FTS5 (stdlib, no extra dependency).

This is a derived index: ChromaDB stays the source of truth and VectorStore keeps the two in
step (and rebuilds this one from ChromaDB if they ever disagree).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path

_FILTER_COLUMNS = frozenset({"source", "document_id"})
_WORD = re.compile(r"\w+")


class KeywordIndex:
    def __init__(self, db_path: Path, table: str) -> None:
        # `table` is derived from a hash, never from user input, so it is safe to interpolate.
        self._table = table
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path)
        # Porter stemming: "indexing", "indexed" and "index" match each other.
        self._db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5("
            "text, chunk_id UNINDEXED, source UNINDEXED, document_id UNINDEXED, "
            "tokenize='porter unicode61')"
        )
        self._db.commit()

    def ids(self) -> set[str]:
        return {row[0] for row in self._db.execute(f"SELECT chunk_id FROM {self._table}")}

    def add(self, rows: Iterable[tuple[str, str, str, str]]) -> None:
        """Insert (chunk_id, source, document_id, text) rows."""
        with self._db:
            self._db.executemany(
                f"INSERT INTO {self._table}(chunk_id, source, document_id, text) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def delete(self, chunk_ids: Iterable[str]) -> None:
        with self._db:
            self._db.executemany(
                f"DELETE FROM {self._table} WHERE chunk_id = ?", [(i,) for i in chunk_ids]
            )

    def clear(self) -> None:
        with self._db:
            self._db.execute(f"DELETE FROM {self._table}")

    def search(self, query: str, limit: int, filters: Mapping[str, str] | None = None) -> list[str]:
        """Chunk ids ranked by BM25, best first. Any query word may match (OR)."""
        words = list(dict.fromkeys(_WORD.findall(query.lower())))
        if not words:
            return []
        conditions = [f"{self._table} MATCH ?"]
        params: list[str | int] = [" OR ".join(f'"{w}"' for w in words)]
        for column, value in (filters or {}).items():
            if column not in _FILTER_COLUMNS:
                raise ValueError(f"keyword search cannot filter on {column!r}")
            conditions.append(f"{column} = ?")
            params.append(value)
        params.append(limit)
        sql = (
            f"SELECT chunk_id FROM {self._table} WHERE {' AND '.join(conditions)} "
            f"ORDER BY bm25({self._table}) LIMIT ?"
        )
        return [row[0] for row in self._db.execute(sql, params)]
