"""Paragraph- and sentence-aware chunking.

Sizes are measured in characters using the common ~4 characters per token
approximation for English text. This avoids a tokenizer dependency; the
trade-off is that chunk sizes in real tokens vary with the language and content.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Iterable

from mini_rag.models import Chunk, Document

CHARS_PER_TOKEN = 4

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
# Capturing, so the whitespace between sentences is kept and line breaks survive.
_SENTENCE_SPLIT = re.compile(r"((?<=[.!?])\s+)")

# (text, what separates it from the unit before: blank line, line break or space)
_Unit = tuple[str, str]


def _join(units: list[_Unit]) -> str:
    return units[0][0] + "".join(separator + text for text, separator in units[1:])


def _split_long(text: str, separator: str, max_chars: int) -> list[_Unit]:
    """Cut an oversized sentence at line breaks, then between words; only a single word longer
    than a chunk is ever cut."""
    units: list[_Unit] = []
    for line in text.split("\n"):
        pieces = textwrap.wrap(
            line.strip(),
            max_chars,
            expand_tabs=False,
            replace_whitespace=False,
            break_on_hyphens=False,
        )
        for number, piece in enumerate(pieces):
            units.append((piece, " " if number else separator))
        if pieces:
            separator = "\n"
    return units


def _split_units(text: str, max_chars: int) -> list[_Unit]:
    """Paragraphs; oversized ones become sentences; oversized sentences become lines or words."""
    units: list[_Unit] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append((paragraph, "\n\n"))
            continue
        separator = "\n\n"
        # Alternates sentence, gap, sentence, ..., with a final empty gap added.
        parts = [*_SENTENCE_SPLIT.split(paragraph), ""]
        for sentence, gap in zip(parts[::2], parts[1::2], strict=True):
            if len(sentence) <= max_chars:
                units.append((sentence, separator))
            else:
                units.extend(_split_long(sentence, separator, max_chars))
            separator = "\n" if "\n" in gap else " "
    return units


def _overlap_tail(units: list[_Unit], overlap_chars: int) -> list[_Unit]:
    tail: list[_Unit] = []
    for unit in reversed(units):
        if len(_join([unit, *tail])) > overlap_chars:
            break
        tail.insert(0, unit)
    return tail


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of roughly `chunk_size` tokens, overlapping by whole units."""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    max_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = chunk_overlap * CHARS_PER_TOKEN

    chunks: list[str] = []
    current: list[_Unit] = []
    for unit in _split_units(text, max_chars):
        if current and len(_join([*current, unit])) > max_chars:
            chunks.append(_join(current))
            current = _overlap_tail(current, overlap_chars)
            # The carried-over overlap must leave room for the new unit.
            while current and len(_join([*current, unit])) > max_chars:
                current.pop(0)
        current.append(unit)
    if current:
        chunks.append(_join(current))
    return chunks


def chunk_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk documents in order. `chunk_index` counts per source document across pages."""
    counters: dict[str, int] = {}
    chunks: list[Chunk] = []
    for document in documents:
        document_id = str(document.metadata.get("document_id", document.metadata.get("source", "")))
        for text in split_text(document.text, chunk_size, chunk_overlap):
            index = counters.get(document_id, 0)
            counters[document_id] = index + 1
            chunks.append(Chunk(text=text, metadata={**document.metadata, "chunk_index": index}))
    return chunks
