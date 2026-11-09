from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import docx
import pymupdf
import pytest

DIM = 64


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors."""

    def __init__(self, model: str = "fake-embed") -> None:
        self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vec = [0.0] * DIM
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            vectors.append(vec if any(vec) else [1.0] + [0.0] * (DIM - 1))
        return vectors


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def make_pdf(path: Path, pages: list[str]) -> None:
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf.save(str(path))
    pdf.close()


def make_docx(path: Path, paragraphs: list[str]) -> None:
    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(str(path))
