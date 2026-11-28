"""Retrieval evaluation with synthetic questions (one per sampled chunk).

Recipe adapted from daveebbelaar/ai-cookbook (hybrid-retrieval/docs/build-your-own-eval.md), but
with a local LLM: each question is generated from one chunk, and that chunk is the single
relevant answer. Scores are only meaningful relative to each other (e.g. before/after a retrieval
change on the *same* frozen question set), not as absolute quality.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from mini_rag.models import Chunk, MiniRagError, RetrievedChunk
from mini_rag.ollama_utils import make_client, ollama_errors
from mini_rag.vector_store import chunk_id

logger = logging.getLogger(__name__)

# Rules follow the cookbook's recipe: answerable from the passage, natural wording rather than
# copying (copied phrasing flatters keyword search), short.
QUESTION_PROMPT = """\
Write one question that a real user might ask and that the passage below answers.

Rules:
- The passage must contain the answer.
- Use your own natural wording; do not copy phrases from the passage.
- Do not mention "the passage" or "the document".
- Maximum 20 words.
- Reply with the question only.

Passage:
{passage}
"""


class EvalError(MiniRagError):
    pass


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    chunk_id: str
    document_id: str
    source: str
    page: int | None = None


@dataclass(frozen=True)
class EvalReport:
    k: int
    questions: int  # scored
    skipped: int  # gold chunk no longer in the index
    chunk_hit: dict[int, float]  # fraction with the gold chunk in the top n
    doc_hit: dict[int, float]  # fraction with any chunk of the gold document in the top n
    mrr: float
    ndcg: float
    misses: list[str]  # questions whose gold chunk was not in the top k


def _clean_question(raw: str) -> str:
    for line in raw.splitlines():
        line = line.strip().strip("`\"'* ")
        if line.lower().startswith("question:"):
            line = line[len("question:") :].strip()
        if line:
            return line
    return ""


def ollama_question_writer(model: str, host: str, num_ctx: int) -> Callable[[str], str]:
    client = make_client(host)

    def write(passage: str) -> str:
        with ollama_errors(model, host):
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": QUESTION_PROMPT.format(passage=passage)}],
                think=False,  # no reasoning needed to write a question; much faster
                options={"num_ctx": num_ctx, "temperature": 0.3},
            )
        return _clean_question(response.message.content or "")

    return write


def generate_questions(
    samples: Iterable[tuple[str, Chunk]], write_question: Callable[[str], str]
) -> list[EvalQuestion]:
    questions = []
    for cid, chunk in samples:
        text = write_question(chunk.text)
        if not text:
            logger.warning("No question generated for chunk %s; skipping", cid)
            continue
        page = chunk.metadata.get("page")
        questions.append(
            EvalQuestion(
                question=text,
                chunk_id=cid,
                document_id=str(chunk.metadata["document_id"]),
                source=str(chunk.metadata.get("source", "")),
                page=page if isinstance(page, int) else None,
            )
        )
    return questions


def write_questions(path: Path, questions: Sequence[EvalQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(q)) + "\n" for q in questions), encoding="utf-8")


def load_questions(path: Path) -> list[EvalQuestion]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvalError(f"Cannot read question file {path}: {exc}") from exc
    try:
        questions = [EvalQuestion(**json.loads(line)) for line in lines if line.strip()]
    except (ValueError, TypeError) as exc:
        raise EvalError(f"{path} is not a valid question file: {exc}") from exc
    if not questions:
        raise EvalError(f"{path} contains no questions")
    return questions


def evaluate(
    questions: Sequence[EvalQuestion],
    retrieve: Callable[[str, int], list[RetrievedChunk]],
    indexed_ids: set[str],
    k: int = 10,
) -> EvalReport:
    cutoffs = sorted({n for n in (1, 3, 5, k) if n <= k})
    chunk_hits = dict.fromkeys(cutoffs, 0)
    doc_hits = dict.fromkeys(cutoffs, 0)
    reciprocal_sum = ndcg_sum = 0.0
    scored = 0
    misses: list[str] = []

    for q in questions:
        if q.chunk_id not in indexed_ids:
            continue  # chunking or content changed since the set was generated
        scored += 1
        results = retrieve(q.question, k)
        chunk_rank = next(
            (i for i, r in enumerate(results, 1) if chunk_id(r.chunk.metadata) == q.chunk_id), None
        )
        doc_rank = next(
            (
                i
                for i, r in enumerate(results, 1)
                if r.chunk.metadata.get("document_id") == q.document_id
            ),
            None,
        )
        for n in cutoffs:
            chunk_hits[n] += chunk_rank is not None and chunk_rank <= n
            doc_hits[n] += doc_rank is not None and doc_rank <= n
        if chunk_rank is None:
            misses.append(q.question)
        else:
            reciprocal_sum += 1 / chunk_rank
            # One relevant chunk, so the ideal DCG is 1.
            ndcg_sum += 1 / math.log2(chunk_rank + 1)

    if scored == 0:
        raise EvalError(
            "None of the questions match chunks in the current index. Re-generate the set "
            "(chunking or file contents changed since it was created)."
        )
    return EvalReport(
        k=k,
        questions=scored,
        skipped=len(questions) - scored,
        chunk_hit={n: c / scored for n, c in chunk_hits.items()},
        doc_hit={n: c / scored for n, c in doc_hits.items()},
        mrr=reciprocal_sum / scored,
        ndcg=ndcg_sum / scored,
        misses=misses,
    )


def format_report(report: EvalReport, max_misses: int = 5) -> str:
    lines = [f"Retrieval eval: {report.questions} questions, k={report.k}"]
    if report.skipped:
        lines.append(f"  ({report.skipped} skipped: gold chunk not in the current index)")
    lines.append("")
    for n, value in report.chunk_hit.items():
        lines.append(f"  chunk hit@{n:<3} {value:6.1%}")
    lines.append(f"  MRR@{report.k:<7} {report.mrr:6.3f}")
    lines.append(f"  NDCG@{report.k:<6} {report.ndcg:6.3f}")
    lines.append("")
    for n, value in report.doc_hit.items():
        lines.append(f"  doc hit@{n:<4} {value:6.1%}")
    if report.misses:
        lines += [
            "",
            f"Missed ({len(report.misses)}), first {min(max_misses, len(report.misses))}:",
        ]
        lines += [f"  - {m}" for m in report.misses[:max_misses]]
    return "\n".join(lines)
