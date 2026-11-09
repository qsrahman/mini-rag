"""Terminal presentation: formatting, spinner, and the prompt_toolkit chat shell.

This module only talks to the RAG service; it contains no retrieval logic.
"""

from __future__ import annotations

import itertools
import logging
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import clear

from mini_rag.config import Settings
from mini_rag.models import Answer, MiniRagError
from mini_rag.rag import RagService

logger = logging.getLogger(__name__)

BANNER = """\
╭──────────────────────────────────────────────╮
│                  MINI RAG                    │
│        Chat with your local documents        │
╰──────────────────────────────────────────────╯"""

HELP_TEXT = """\
Commands:
  /help     Show this help
  /status   Show model, database and index status
  /clear    Clear the screen
  /exit     Exit (also /quit, Ctrl+D)

Keys:
  Enter        Send the question
  Alt+Enter    Insert a newline (also Ctrl+J); pasted multi-line text is kept as is
  Up / Down    Browse history          Ctrl+R  Search history
  Ctrl+C       Cancel current input or running answer"""


def format_answer(answer: Answer) -> str:
    lines = [answer.text]
    if answer.sources:
        lines += ["", "Sources:"]
        for source in answer.sources:
            page = f" (page {source.page})" if source.page else ""
            lines.append(f"  [{source.id}] {source.name}{page}")
    if answer.unknown_citations:
        cited = ", ".join(answer.unknown_citations)
        lines += [
            "",
            f"Warning: the answer cites {cited}, which was not among the retrieved sources.",
        ]
    return "\n".join(lines)


@contextmanager
def working(message: str) -> Iterator[None]:
    """Show a spinner on stderr while a blocking call runs (only on a real terminal)."""
    if not sys.stderr.isatty():
        yield
        return
    stop = threading.Event()

    def spin() -> None:
        for frame in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            sys.stderr.write(f"\r{frame} {message}")
            sys.stderr.flush()
            if stop.wait(0.1):
                break

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        sys.stderr.write("\r\x1b[2K")
        sys.stderr.flush()


def create_session(history_path: Path) -> PromptSession[str]:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    bindings = KeyBindings()

    @bindings.add("escape", "enter")  # Alt+Enter
    @bindings.add("c-j")
    def _newline(event) -> None:  # type: ignore[no-untyped-def]
        event.current_buffer.insert_text("\n")

    return PromptSession(
        history=FileHistory(str(history_path)),
        key_bindings=bindings,
        enable_history_search=True,
        prompt_continuation="    ",
        bottom_toolbar="Enter: send · Alt+Enter: newline · /help",
    )


class ChatShell:
    def __init__(
        self,
        rag: RagService,
        settings: Settings,
        chunk_count: Callable[[], int],
        session: PromptSession[str] | None = None,
        write: Callable[[str, str], None] | None = None,
    ) -> None:
        self._rag = rag
        self._settings = settings
        self._chunk_count = chunk_count
        self._session = session or create_session(settings.history_path)
        self._write = write or self._default_write
        self._prompt = FormattedText([("bold ansigreen", "You > ")])

    @staticmethod
    def _default_write(text: str, style: str = "") -> None:
        print_formatted_text(FormattedText([(style, text)]))

    def status_text(self) -> str:
        s = self._settings
        return (
            f"Model: {s.llm_model}\n"
            f"Embedding: {s.embedding_model}\n"
            f"Ollama: {s.ollama_host}\n"
            f"Database: {s.vector_db_path}\n"
            f"Collection: {s.collection_name}\n"
            f"Indexed chunks: {self._chunk_count()}"
        )

    def run(self) -> int:
        self._write(BANNER, "bold ansicyan")
        self._write("", "")
        self._write(self.status_text(), "")
        self._write(
            "\nType /help for available commands.\nType /exit to quit.\n", "ansibrightblack"
        )
        while True:
            try:
                line = self._session.prompt(self._prompt)
            except KeyboardInterrupt:
                self._write("(Input cancelled. Use /exit or Ctrl+D to quit.)", "ansibrightblack")
                continue
            except EOFError:
                break
            if not self.handle(line):
                break
        self._write("Goodbye.", "")
        return 0

    def handle(self, line: str) -> bool:
        """Process one input line. Returns False when the shell should exit."""
        line = line.strip()
        if not line:
            return True
        if line.startswith("/"):
            return self._command(line)
        self._ask(line)
        return True

    def _command(self, line: str) -> bool:
        name = line.split()[0].lower()
        if name in ("/exit", "/quit"):
            return False
        if name == "/help":
            self._write(HELP_TEXT, "")
        elif name == "/status":
            self._write(self.status_text(), "")
        elif name == "/clear":
            clear()
        else:
            self._write(f"Unknown command {name!r}. Type /help for available commands.", "ansired")
        return True

    def _ask(self, question: str) -> None:
        try:
            with working("Searching documents and generating an answer (Ctrl+C to cancel)..."):
                answer = self._rag.ask(question)
        except KeyboardInterrupt:
            self._write("Cancelled.", "ansibrightblack")
            return
        except MiniRagError as exc:
            self._write(f"Error: {exc}", "ansired")
            return
        except Exception as exc:  # keep the shell alive; details go to the log
            logger.exception("Unexpected error while answering")
            self._write(f"Unexpected error: {exc}", "ansired")
            return
        self._write("\nAssistant:", "bold ansiyellow")
        self._write(format_answer(answer) + "\n", "")
