"""Shared translation of Ollama SDK failures into readable errors."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import ollama

from mini_rag.models import MiniRagError


class OllamaError(MiniRagError):
    pass


@contextmanager
def ollama_errors(model: str, host: str) -> Iterator[None]:
    try:
        yield
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise OllamaError(
                f"Ollama model {model!r} is not available. Run: ollama pull {model}"
            ) from exc
        raise OllamaError(f"Ollama returned an error for model {model!r}: {exc}") from exc
    except ConnectionError as exc:
        raise OllamaError(
            f"Cannot connect to Ollama at {host}. Is it running? (start it with: ollama serve)"
        ) from exc
