import pytest

from mini_rag.config import ConfigError, Settings


def test_defaults():
    s = Settings.from_env({})
    assert (s.llm_model, s.embedding_model) == ("qwen3.5:9b-mlx", "nomic-embed-text-v2-moe")
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (400, 60, 5)
    assert "document question-answering assistant" in s.system_prompt


def test_environment_overrides():
    s = Settings.from_env({"RAG_TOP_K": "9", "RAG_COLLECTION_NAME": "c", "RAG_CHUNK_SIZE": "200"})
    assert (s.top_k, s.collection_name, s.chunk_size) == (9, "c", 200)


@pytest.mark.parametrize(
    "env",
    [
        {"RAG_CHUNK_SIZE": "0"},
        {"RAG_CHUNK_OVERLAP": "800"},
        {"RAG_CHUNK_OVERLAP": "-1"},
        {"RAG_TOP_K": "0"},
        {"RAG_TOP_K": "many"},
        {"RAG_LLM_MODEL": " "},
    ],
)
def test_invalid_values_rejected(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_cli_overrides_skip_none_and_are_validated():
    s = Settings().with_overrides(top_k=3, llm_model=None)
    assert s.top_k == 3 and s.llm_model == "qwen3.5:9b-mlx"
    with pytest.raises(ConfigError):
        Settings().with_overrides(chunk_overlap=5000)


def test_retrieval_and_rerank_settings():
    s = Settings.from_env({})
    assert (s.retrieval, s.rerank) == ("hybrid", False)
    s = Settings.from_env({"RAG_RETRIEVAL": "dense", "RAG_RERANK": "True"})
    assert (s.retrieval, s.rerank) == ("dense", True)
    with pytest.raises(ConfigError, match="retrieval"):
        Settings.from_env({"RAG_RETRIEVAL": "sparse"})
