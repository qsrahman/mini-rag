from unittest.mock import MagicMock

import pytest
from prompt_toolkit.history import FileHistory

from mini_rag.config import Settings
from mini_rag.interactive import ChatShell, create_session, format_answer
from mini_rag.models import Answer, MiniRagError, Source


class Harness:
    def __init__(self, tmp_path, inputs=()):
        self.rag = MagicMock()
        self.rag.ask.return_value = Answer("Answer text [S1]", [Source("S1", "ml.pdf", 12)])
        self.output: list[str] = []
        self.session = MagicMock()
        self.session.prompt.side_effect = list(inputs)
        self.settings = Settings(history_path=tmp_path / "h")
        self.shell = ChatShell(
            self.rag,
            self.settings,
            chunk_count=lambda: 245,
            session=self.session,
            write=lambda text, style="": self.output.append(text),
        )

    @property
    def text(self):
        return "\n".join(self.output)


def test_format_answer_lists_sources_and_warnings():
    answer = Answer("Body", [Source("S1", "a.pdf", 2), Source("S2", "b.docx")], ["S5"])
    out = format_answer(answer)
    assert "  [S1] a.pdf (page 2)" in out
    assert "  [S2] b.docx\n" in out + "\n"
    assert "S5" in out and "Warning" in out


def test_format_answer_without_sources():
    assert format_answer(Answer("None found", [])) == "None found"


def test_help_and_unknown_commands(tmp_path):
    h = Harness(tmp_path)
    assert h.shell.handle("/help")
    assert "/exit" in h.text
    assert h.shell.handle("/bogus")
    assert "Unknown command" in h.text
    h.rag.ask.assert_not_called()


@pytest.mark.parametrize("command", ["/exit", "/quit", " /EXIT "])
def test_exit_commands(tmp_path, command):
    assert Harness(tmp_path).shell.handle(command) is False


def test_status_shows_models_and_chunk_count(tmp_path):
    h = Harness(tmp_path)
    h.shell.handle("/status")
    assert "qwen3.5:9b-mlx" in h.text
    assert "nomic-embed-text-v2-moe" in h.text
    assert "Indexed chunks: 245" in h.text


def test_question_is_delegated_to_rag(tmp_path):
    h = Harness(tmp_path)
    assert h.shell.handle("  What is ML?  ")
    h.rag.ask.assert_called_once_with("What is ML?")
    assert "Answer text" in h.text and "[S1] ml.pdf (page 12)" in h.text


def test_blank_input_is_ignored(tmp_path):
    h = Harness(tmp_path)
    assert h.shell.handle("   ")
    h.rag.ask.assert_not_called()


def test_rag_errors_are_reported_and_shell_continues(tmp_path):
    h = Harness(tmp_path)
    h.rag.ask.side_effect = MiniRagError("Ollama is down")
    assert h.shell.handle("q")
    assert "Error: Ollama is down" in h.text


def test_unexpected_errors_do_not_crash_shell(tmp_path):
    h = Harness(tmp_path)
    h.rag.ask.side_effect = RuntimeError("kaboom")
    assert h.shell.handle("q")
    assert "Unexpected error: kaboom" in h.text


def test_interrupting_inference_cancels_gracefully(tmp_path):
    h = Harness(tmp_path)
    h.rag.ask.side_effect = KeyboardInterrupt
    assert h.shell.handle("q")
    assert "Cancelled" in h.text


def test_ctrl_d_exits_the_loop(tmp_path):
    h = Harness(tmp_path, inputs=[EOFError])
    assert h.shell.run() == 0
    assert "Goodbye" in h.text


def test_ctrl_c_at_prompt_keeps_running(tmp_path):
    h = Harness(tmp_path, inputs=[KeyboardInterrupt, "hello", "/exit"])
    assert h.shell.run() == 0
    assert "Input cancelled" in h.text
    h.rag.ask.assert_called_once_with("hello")


def test_run_prints_banner_and_status(tmp_path):
    h = Harness(tmp_path, inputs=["/exit"])
    h.shell.run()
    assert "MINI RAG" in h.text and "Indexed chunks: 245" in h.text


def test_session_uses_persistent_file_history(tmp_path):
    path = tmp_path / "nested" / "history"
    session = create_session(path)
    assert isinstance(session.history, FileHistory)
    assert session.history.filename == str(path)
    assert path.parent.is_dir()
