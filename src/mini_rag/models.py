"""Plain data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

# ChromaDB metadata values must be scalars; we only use str and int.
Metadata = dict[str, str | int]


class MiniRagError(Exception):
    """Base class for expected, user-reportable errors."""


@dataclass(frozen=True)
class Document:
    """Extracted text plus where it came from (one PDF page, or a whole file)."""

    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: Metadata = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable and unique: changes whenever the file content or chunking changes."""
        m = self.metadata
        return f"{m['document_id']}-{str(m['content_hash'])[:12]}-{m['chunk_index']}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    # Cosine distance of the dense match (0 = identical); None if only keyword search found it.
    distance: float | None = None


@dataclass(frozen=True)
class Source:
    """A retrieved chunk as presented to the user, labelled [S1], [S2], ..."""

    id: str
    name: str
    page: int | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    # Always the sources that were actually retrieved, never what the LLM claims.
    sources: list[Source]
    # Citation ids in the answer text that do not match any retrieved source.
    unknown_citations: list[str] = field(default_factory=list)
