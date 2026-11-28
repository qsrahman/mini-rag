"""Optional reranking of retrieved chunks with the local chat model.

Ollama has no rerank endpoint, so each candidate is scored with a yes/no relevance question and
the probability mass on "yes" vs "no" (from token logprobs) is the score, the same idea as the
Qwen3-Reranker prompt format. One LLM call per candidate: it improves ordering but adds
roughly a second per candidate.
"""

from __future__ import annotations

import math

import ollama

from mini_rag.models import RetrievedChunk
from mini_rag.ollama_utils import OllamaError, make_client, ollama_errors

_SYSTEM = (
    "Judge whether the Document is relevant to and answers the Query. "
    'Note that the answer can only be "yes" or "no".'
)


class LlmReranker:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        num_ctx: int = 4096,
        client: ollama.Client | None = None,
    ) -> None:
        self.model = model
        self._host = host
        self._num_ctx = num_ctx
        self._client = client or make_client(host)

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        scores = [self._score(query, c.chunk.text) for c in candidates]
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])  # stable on ties
        return [candidates[i] for i in order]

    def _score(self, query: str, document: str) -> float:
        with ollama_errors(self.model, self._host):
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"<Query>: {query}\n<Document>: {document}"},
                ],
                think=False,  # a one-token verdict needs no reasoning
                logprobs=True,
                top_logprobs=10,
                options={"num_predict": 1, "temperature": 0, "num_ctx": self._num_ctx},
            )
        if not response.logprobs:
            raise OllamaError(f"Model {self.model!r} did not return token logprobs; cannot rerank")
        yes = no = 0.0
        for alternative in response.logprobs[0].top_logprobs or []:
            token = alternative.token.strip().lower()
            if token == "yes":
                yes += math.exp(alternative.logprob)
            elif token == "no":
                no += math.exp(alternative.logprob)
        return yes / (yes + no) if yes + no > 0 else 0.0
