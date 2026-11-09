"""Live end-to-end tests against a real local Ollama. Run with: uv run pytest -m live

Skipped automatically when Ollama or the default models are unavailable.
Assertions on LLM text are deliberately loose; model output is not deterministic.
"""

from __future__ import annotations

import os
import pty
import re
import select
import sys
import time
from pathlib import Path

import ollama
import pytest

from mini_rag.cli import build_services
from mini_rag.config import Settings
from mini_rag.indexer import Indexer
from tests.conftest import make_docx, make_pdf

pytestmark = pytest.mark.live

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[=>]")


def _require_ollama(settings: Settings) -> None:
    try:
        installed = {m.model for m in ollama.Client(host=settings.ollama_host).list().models}
    except Exception as exc:
        pytest.skip(f"Ollama not reachable: {exc}")
    for needed in (settings.llm_model, settings.embedding_model):
        if not any(name == needed or name.split(":")[0] == needed for name in installed if name):
            pytest.skip(f"Ollama model {needed!r} not installed")


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    base = tmp_path_factory.mktemp("live")
    docs = base / "docs"
    docs.mkdir()
    (docs / "geography.txt").write_text("The capital of Freedonia is Fredville.\n")
    (docs / "food.md").write_text("# Food\n\nGlorp is the national dish of Sylvania.\n")
    make_docx(docs / "mountains.docx", ["The Quibble reactor runs on purple sunlight."])
    make_pdf(docs / "peaks.pdf", ["Intro page about nothing.", "Mount Zork is 4000 metres tall."])

    settings = Settings(vector_db_path=base / "chroma", history_path=base / "history")
    _require_ollama(settings)
    services = build_services(settings)
    summary = Indexer(
        services.embedder, services.store, settings.chunk_size, settings.chunk_overlap
    ).index_directory(docs)
    assert summary.failed == []
    assert summary.indexed == 4
    return settings, services, docs, base


@pytest.mark.parametrize(
    "question,expected_source,expected_page,expected_text",
    [
        ("What is the capital of Freedonia?", "geography.txt", None, "fredville"),
        ("What is the national dish of Sylvania?", "food.md", None, "glorp"),
        ("What does the Quibble reactor run on?", "mountains.docx", None, "sunlight"),
        ("How tall is Mount Zork?", "peaks.pdf", 2, "4000"),
    ],
)
def test_answers_are_grounded_and_cited(
    live, question, expected_source, expected_page, expected_text
):
    _, services, _, _ = live
    answer = services.rag.ask(question, top_k=3)
    assert expected_text in answer.text.lower().replace(",", "")
    match = [s for s in answer.sources if s.name == expected_source]
    assert match, f"{expected_source} not among retrieved sources: {answer.sources}"
    if expected_page:
        assert match[0].page == expected_page
    assert answer.unknown_citations == []


def test_unanswerable_question_is_declined(live):
    _, services, _, _ = live
    answer = services.rag.ask("Who won the 1998 football World Cup?", top_k=3)
    assert "france" not in answer.text.lower()
    assert re.search(
        r"cannot|can't|not (found|contain|mention|provide)|no information|unable",
        answer.text.lower(),
    )


def test_second_index_run_is_a_no_op(live):
    settings, services, docs, _ = live
    summary = Indexer(
        services.embedder, services.store, settings.chunk_size, settings.chunk_overlap
    ).index_directory(docs)
    assert (summary.indexed, summary.unchanged) == (0, 4)


# --- Ctrl+C during a running generation (real terminal via a pty) ---


class PtyProcess:
    def __init__(self, args: list[str], env: dict[str, str]) -> None:
        self.pid, self.fd = pty.fork()
        if self.pid == 0:  # child
            os.execve(sys.executable, [sys.executable, "-m", "mini_rag", *args], env)
        self._text = ""
        self._consumed = 0

    def send(self, data: bytes) -> None:
        os.write(self.fd, data)

    def wait_for(self, needle: str, timeout: float) -> bool:
        """Wait for `needle` to appear in output not yet consumed by an earlier wait."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = self._text.find(needle, self._consumed)
            if found >= 0:
                self._consumed = found + len(needle)
                return True
            if select.select([self.fd], [], [], 0.2)[0]:
                try:
                    chunk = os.read(self.fd, 65536)
                except OSError:
                    return False
                self._text += _ANSI.sub("", chunk.decode(errors="replace"))
        return False

    def exit_code(self, timeout: float) -> int | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                return os.waitstatus_to_exitcode(status)
            time.sleep(0.1)
        return None

    def kill(self) -> None:
        try:
            os.kill(self.pid, 9)
            os.waitpid(self.pid, 0)
        except ProcessLookupError, ChildProcessError:
            pass
        os.close(self.fd)


def _env(settings: Settings, base: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")}
    env.update(
        RAG_VECTOR_DB_PATH=str(settings.vector_db_path),
        RAG_HISTORY_PATH=str(settings.history_path),
        PYTHONWARNINGS="ignore",
    )
    return env


def test_ctrl_c_cancels_generation_in_chat_and_shell_survives(live):
    settings, _, _, base = live
    proc = PtyProcess(["chat"], _env(settings, base))
    try:
        assert proc.wait_for("You >", 90), proc._text
        proc.send(b"Explain everything you know about Mount Zork in great detail.\r")
        assert proc.wait_for("Searching documents", 30), proc._text
        time.sleep(1.5)  # let it get past embedding into generation
        proc.send(b"\x03")
        assert proc.wait_for("Cancelled.", 20), proc._text
        assert proc.wait_for("You >", 10), proc._text  # prompt is back and responsive
        proc.send(b"/status\r")
        assert proc.wait_for("Indexed chunks:", 10), proc._text
        assert proc.wait_for("You >", 10), proc._text  # wait for the prompt before typing again
        proc.send(b"/exit\r")
        assert proc.wait_for("Goodbye.", 10), proc._text
        assert proc.exit_code(15) == 0
    finally:
        proc.kill()


def test_ctrl_c_during_one_shot_ask_exits_130(live):
    settings, _, _, base = live
    proc = PtyProcess(
        ["ask", "Explain everything about Mount Zork in great detail."], _env(settings, base)
    )
    try:
        assert proc.wait_for("Searching documents", 60), proc._text
        time.sleep(1.5)
        proc.send(b"\x03")
        assert proc.wait_for("Interrupted.", 20), proc._text
        assert proc.exit_code(15) == 130
    finally:
        proc.kill()
