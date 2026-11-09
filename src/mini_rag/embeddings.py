"""Embedding generation. Independent of the vector database."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import ollama

from mini_rag.ollama_utils import OllamaError, ollama_errors

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns texts into vectors. Swap implementations freely."""

    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        batch_size: int = 32,
        client: ollama.Client | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model = model
        self._host = host
        self._batch_size = batch_size
        self._client = client or ollama.Client(host=host)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            with ollama_errors(self.model, self._host):
                response = self._client.embed(model=self.model, input=batch)
            if len(response.embeddings) != len(batch):
                raise OllamaError(
                    f"Ollama returned {len(response.embeddings)} embeddings for {len(batch)} inputs"
                )
            vectors.extend([list(v) for v in response.embeddings])
        return vectors
