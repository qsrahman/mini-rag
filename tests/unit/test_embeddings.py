from types import SimpleNamespace
from unittest.mock import MagicMock

import ollama
import pytest

from mini_rag.embeddings import OllamaEmbedder
from mini_rag.ollama_utils import OllamaError


def client_returning_one_vector_per_input():
    client = MagicMock()
    client.embed.side_effect = lambda model, input: SimpleNamespace(
        embeddings=[[float(len(t)), 1.0] for t in input]
    )
    return client


def test_single_text():
    client = client_returning_one_vector_per_input()
    assert OllamaEmbedder("m", client=client).embed(["abc"]) == [[3.0, 1.0]]
    client.embed.assert_called_once_with(model="m", input=["abc"])


def test_batches_preserve_order():
    client = client_returning_one_vector_per_input()
    vectors = OllamaEmbedder("m", batch_size=2, client=client).embed(
        ["a", "bb", "ccc", "dddd", "e"]
    )
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 1.0]
    assert [c.kwargs["input"] for c in client.embed.call_args_list] == [
        ["a", "bb"],
        ["ccc", "dddd"],
        ["e"],
    ]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_is_rejected(batch_size):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        OllamaEmbedder("m", batch_size=batch_size)


def test_empty_input_makes_no_calls():
    client = MagicMock()
    assert OllamaEmbedder("m", client=client).embed([]) == []
    client.embed.assert_not_called()


def test_connection_failure():
    client = MagicMock()
    client.embed.side_effect = ConnectionError("refused")
    with pytest.raises(OllamaError, match="Cannot connect to Ollama"):
        OllamaEmbedder("m", host="http://x:1", client=client).embed(["a"])


def test_missing_model_suggests_pull():
    client = MagicMock()
    client.embed.side_effect = ollama.ResponseError("not found", 404)
    with pytest.raises(OllamaError, match="ollama pull m"):
        OllamaEmbedder("m", client=client).embed(["a"])


def test_other_response_errors_are_wrapped():
    client = MagicMock()
    client.embed.side_effect = ollama.ResponseError("boom", 500)
    with pytest.raises(OllamaError, match="boom"):
        OllamaEmbedder("m", client=client).embed(["a"])


def test_wrong_number_of_embeddings():
    client = MagicMock()
    client.embed.return_value = SimpleNamespace(embeddings=[[1.0]])
    with pytest.raises(OllamaError, match="1 embeddings for 2 inputs"):
        OllamaEmbedder("m", client=client).embed(["a", "b"])
