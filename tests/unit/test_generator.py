from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mini_rag.generator import Generator
from mini_rag.ollama_utils import OllamaError


def reply(content):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def test_generate_sends_system_prompt_context_and_question():
    client = MagicMock()
    client.chat.return_value = reply("  the answer  ")
    generator = Generator("llm", system_prompt="SYS", num_ctx=4096, client=client)
    assert generator.generate("Why?", "[S1] ctx") == "the answer"
    kwargs = client.chat.call_args.kwargs
    assert kwargs["model"] == "llm"
    assert kwargs["options"] == {"num_ctx": 4096}
    system, user = kwargs["messages"]
    assert system == {"role": "system", "content": "SYS"}
    assert "<documents>\n[S1] ctx\n</documents>" in user["content"]
    assert user["content"].endswith("Question: Why?")


def test_connection_error():
    client = MagicMock()
    client.chat.side_effect = ConnectionError()
    with pytest.raises(OllamaError, match="Cannot connect"):
        Generator("llm", client=client).generate("q", "c")


def test_empty_response_is_an_error():
    client = MagicMock()
    client.chat.return_value = reply("")
    with pytest.raises(OllamaError, match="empty"):
        Generator("llm", client=client).generate("q", "c")
