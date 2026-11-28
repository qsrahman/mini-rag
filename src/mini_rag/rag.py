"""RAG orchestration: retrieve -> build context -> generate -> attribute sources."""

from __future__ import annotations

import re
from typing import Any

from mini_rag.generator import Generator
from mini_rag.models import Answer, RetrievedChunk, Source
from mini_rag.retriever import Retriever

NO_CONTENT_MESSAGE = (
    "No indexed content was found for this question, so it cannot be answered from the "
    "available documents. Make sure your documents have been indexed."
)

_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_SOURCE_ID = re.compile(r"\bS(\d+)\b")


def build_sources(retrieved: list[RetrievedChunk]) -> list[Source]:
    sources = []
    for number, item in enumerate(retrieved, start=1):
        page = item.chunk.metadata.get("page")
        sources.append(
            Source(
                id=f"S{number}",
                name=str(item.chunk.metadata.get("source", "unknown")),
                page=page if isinstance(page, int) else None,
            )
        )
    return sources


def build_context(retrieved: list[RetrievedChunk], sources: list[Source]) -> str:
    """Documents as XML-tagged blocks with their source ID and location (Anthropic's long-context
    recommendation); the system prompt tells the model to cite by these IDs."""
    blocks = []
    for item, source in zip(retrieved, sources, strict=True):
        location = f"{source.name}, page {source.page}" if source.page else source.name
        blocks.append(
            f'<document id="{source.id}">\n<source>{location}</source>\n'
            f"<document_content>\n{item.chunk.text}\n</document_content>\n</document>"
        )
    return "\n".join(blocks)


def find_unknown_citations(answer: str, sources: list[Source]) -> list[str]:
    """Citation ids in the answer that don't correspond to a retrieved source."""
    known = {s.id for s in sources}
    unknown: list[str] = []
    for bracket in _BRACKET.findall(answer):
        for number in _SOURCE_ID.findall(bracket):
            citation = f"S{number}"
            if citation not in known and citation not in unknown:
                unknown.append(citation)
    return unknown


class RagService:
    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self._retriever = retriever
        self._generator = generator

    def ask(self, question: str, where: dict[str, Any] | None = None) -> Answer:
        retrieved = self._retriever.retrieve(question, where=where)
        if not retrieved:
            return Answer(text=NO_CONTENT_MESSAGE, sources=[])
        sources = build_sources(retrieved)
        text = self._generator.generate(question, build_context(retrieved, sources))
        return Answer(
            text=text, sources=sources, unknown_citations=find_unknown_citations(text, sources)
        )
