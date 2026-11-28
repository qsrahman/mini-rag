"""BM25 keyword search over chunk text using SQLite FTS5 (stdlib, no extra dependency).

This is a derived index: ChromaDB stays the source of truth and VectorStore keeps the two in
step (and rebuilds this one from ChromaDB if they ever disagree).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from mini_rag.models import MiniRagError

_FILTER_COLUMNS = frozenset({"source", "document_id"})
_WORD = re.compile(r"\w+")

# English function words. Left in an OR query they match almost every chunk and add noise to the
# BM25 ranking. Only function words belong here: "file" or "find" can be what the user is after.
_STOPWORDS = frozenset(
    """a about an and any are at be can do does for from how i in is it its me my of on or
    some that the these this those to was were what when where which who with""".split()  # noqa: SIM905 (far more readable than a list literal)
)


class KeywordIndexError(MiniRagError):
    pass


class KeywordIndex:
    def __init__(self, db_path: Path, table: str) -> None:
        # `table` is derived from a hash, never from user input, so it is safe to interpolate.
        self._table = table
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._errors():
            self._db = sqlite3.connect(db_path)
            # Porter stemming: "indexing", "indexed" and "index" match each other.
            self._db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5("
                "text, chunk_id UNINDEXED, source UNINDEXED, document_id UNINDEXED, "
                "tokenize='porter unicode61')"
            )
            self._db.commit()

    @contextmanager
    def _errors(self) -> Iterator[None]:
        try:
            yield
        except sqlite3.Error as exc:
            if "locked" in str(exc) or "busy" in str(exc):
                raise KeywordIndexError(
                    "The keyword index is busy (is another `mini-rag index` running?). "
                    "Try again in a moment."
                ) from exc
            raise KeywordIndexError(
                f"Keyword index error in {self._path}: {exc}. If it is damaged, delete that "
                "file and run the command again; it is rebuilt automatically."
            ) from exc

    def ids(self) -> set[str]:
        with self._errors():
            return {row[0] for row in self._db.execute(f"SELECT chunk_id FROM {self._table}")}

    def data_version(self) -> int:
        """Changes whenever another connection (i.e. another process) commits to the database."""
        with self._errors():
            version: int = self._db.execute("PRAGMA data_version").fetchone()[0]
        return version

    def row_count(self) -> int:
        with self._errors():
            count: int = self._db.execute(f"SELECT count(*) FROM {self._table}").fetchone()[0]
        return count

    # Each write is one transaction: another process must never see, or add to, a half-done
    # change, otherwise rows get duplicated.

    def add(self, rows: Iterable[tuple[str, str, str, str]]) -> None:
        """Insert (chunk_id, source, document_id, text) rows."""
        with self._errors(), self._db:
            self._insert(rows)

    def replace(self, old_ids: Iterable[str], rows: Iterable[tuple[str, str, str, str]]) -> None:
        """Delete `old_ids` and insert `rows` atomically."""
        with self._errors(), self._db:
            self._delete(old_ids)
            self._insert(rows)

    def rebuild(self, rows: Iterable[tuple[str, str, str, str]]) -> None:
        """Replace the whole index atomically."""
        with self._errors(), self._db:
            self._db.execute(f"DELETE FROM {self._table}")
            self._insert(rows)

    def delete(self, chunk_ids: Iterable[str]) -> None:
        with self._errors(), self._db:
            self._delete(chunk_ids)

    def clear(self) -> None:
        self.rebuild([])

    def _insert(self, rows: Iterable[tuple[str, str, str, str]]) -> None:
        self._db.executemany(
            f"INSERT INTO {self._table}(chunk_id, source, document_id, text) VALUES (?, ?, ?, ?)",
            rows,
        )

    def _delete(self, chunk_ids: Iterable[str]) -> None:
        self._db.executemany(
            f"DELETE FROM {self._table} WHERE chunk_id = ?", [(i,) for i in chunk_ids]
        )

    def search(self, query: str, limit: int, filters: Mapping[str, str] | None = None) -> list[str]:
        """Chunk ids ranked by BM25, best first. Any query word may match (OR).

        Stopwords are ignored unless the query consists of nothing else.
        """
        every_word = list(dict.fromkeys(_WORD.findall(query.lower())))
        words = [w for w in every_word if w not in _STOPWORDS] or every_word
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
        with self._errors():
            return [row[0] for row in self._db.execute(sql, params)]
