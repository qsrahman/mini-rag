"""Answer generation with a local Ollama chat model."""

from __future__ import annotations

import ollama

from mini_rag.config import DEFAULT_SYSTEM_PROMPT
from mini_rag.ollama_utils import OllamaError, ollama_errors


class Generator:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        num_ctx: int = 8192,
        client: ollama.Client | None = None,
    ) -> None:
        self.model = model
        self._host = host
        self._system_prompt = system_prompt
        self._num_ctx = num_ctx
        self._client = client or ollama.Client(host=host)

    def generate(self, question: str, context: str) -> str:
        # Documents first, question last (Anthropic: query at the end improves quality).
        user_message = f"<documents>\n{context}\n</documents>\n\nQuestion: {question}"
        with ollama_errors(self.model, self._host):
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options={"num_ctx": self._num_ctx},
            )
        answer = (response.message.content or "").strip()
        if not answer:
            raise OllamaError(f"Model {self.model!r} returned an empty response")
        return answer
