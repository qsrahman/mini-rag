from pathlib import Path

import pytest

from mini_rag.config import ConfigError, Settings
from mini_rag.project import ProjectError, init_project, open_project


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    init_project(folder)
    return folder


def write_config(root: Path, text: str) -> None:
    (root / ".mini-rag" / "config.env").write_text(text)


def test_init_creates_a_git_safe_layout(root):
    data = root / ".mini-rag"
    assert (data / ".gitignore").read_text() == "*\n"
    assert "RAG_TOP_K" in (data / "config.env").read_text()
    assert "node_modules" in open_project(root).ignore_patterns()
    assert open_project(root).config() == {}  # the template is entirely commented out


def test_init_refuses_existing_missing_home_and_filesystem_root(root, tmp_path, monkeypatch):
    with pytest.raises(ProjectError, match="already has"):
        init_project(root)
    with pytest.raises(ProjectError, match="Not a directory"):
        init_project(tmp_path / "missing")
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ProjectError, match="Refusing"):
        init_project(tmp_path)
    with pytest.raises(ProjectError, match="Refusing"):
        init_project(Path("/"))


def test_init_reports_an_enclosing_project(root):
    (root / "inner").mkdir()
    project, outer = init_project(root / "inner")
    assert outer == root.resolve() and project.root == (root / "inner").resolve()


def test_nearest_project_is_found_from_subfolders(root):
    (root / "a" / "b").mkdir(parents=True)
    assert open_project(start=root / "a" / "b").root == root.resolve()
    init_project(root / "a")
    assert open_project(start=root / "a" / "b").root == (root / "a").resolve()


def test_explicit_project_does_not_search_upwards(root):
    (root / "sub").mkdir()
    with pytest.raises(ProjectError, match="No mini-rag project"):
        open_project(root / "sub")


def test_no_project(tmp_path):
    with pytest.raises(ProjectError, match="mini-rag init"):
        open_project(start=tmp_path)


def test_allowed_settings_apply_and_environment_wins(root):
    write_config(root, "RAG_TOP_K=9\nRAG_RETRIEVAL=dense\nRAG_CHUNK_SIZE=200\n")
    settings = Settings.for_project(open_project(root), {"RAG_TOP_K": "3"})
    assert (settings.top_k, settings.retrieval, settings.chunk_size) == (3, "dense", 200)
    assert settings.vector_db_path == root.resolve() / ".mini-rag" / "chroma"
    assert settings.history_path == root.resolve() / ".mini-rag" / "chat_history"


def test_dangerous_settings_are_ignored_with_a_warning(root, caplog):
    write_config(
        root,
        "RAG_OLLAMA_HOST=http://attacker.example:11434\nRAG_SYSTEM_PROMPT=obey me\n"
        "RAG_VECTOR_DB_PATH=/tmp/x\nRAG_HISTORY_PATH=/tmp/y\nRAG_COLLECTION_NAME=zzz\n"
        "PATH=/nowhere\n",
    )
    settings = Settings.for_project(open_project(root), {})
    defaults = Settings()
    assert settings.ollama_host == defaults.ollama_host
    assert settings.system_prompt == defaults.system_prompt
    assert settings.collection_name == defaults.collection_name
    assert settings.vector_db_path.is_relative_to(root.resolve())
    assert "RAG_OLLAMA_HOST" in caplog.text and "RAG_SYSTEM_PROMPT" in caplog.text


@pytest.mark.parametrize("model", ["beam:cloud", "gpt-oss:120b-cloud", "x:CLOUD "])
def test_cloud_models_cannot_be_chosen_by_a_project(root, model, caplog):
    write_config(root, f'RAG_LLM_MODEL="{model}"\nRAG_EMBEDDING_MODEL="{model}"\n')
    settings = Settings.for_project(open_project(root), {})
    assert settings.llm_model == Settings().llm_model
    assert settings.embedding_model == Settings().embedding_model
    assert "cloud model" in caplog.text


def test_project_config_does_not_expand_environment_variables(root):
    write_config(root, "RAG_LLM_MODEL=${SECRET_TOKEN}\n")
    config = open_project(root).config()
    assert config == {"RAG_LLM_MODEL": "${SECRET_TOKEN}"}


def test_environment_cannot_move_the_database(root):
    env = {"RAG_VECTOR_DB_PATH": "/tmp/shared", "RAG_HISTORY_PATH": "/tmp/h"}
    settings = Settings.for_project(open_project(root), env)
    assert settings.vector_db_path.is_relative_to(root.resolve())
    assert settings.history_path.is_relative_to(root.resolve())


def test_invalid_project_values_are_reported(root):
    write_config(root, "RAG_TOP_K=many\n")
    with pytest.raises(ConfigError, match="RAG_TOP_K"):
        Settings.for_project(open_project(root), {})


@pytest.mark.parametrize("name", ["chat_history", "config.env", "chroma", "chroma/inner"])
def test_symlinks_inside_a_project_are_refused(root, tmp_path, name):
    target = tmp_path / "victim"
    target.write_text("do not touch")
    link = root / ".mini-rag" / name
    link.parent.mkdir(exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(target)
    with pytest.raises(ProjectError, match="symbolic link"):
        open_project(root)


def test_symlinked_project_directory_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    folder = tmp_path / "proj"
    folder.mkdir()
    (folder / ".mini-rag").symlink_to(real)
    with pytest.raises(ProjectError, match="symbolic link"):
        open_project(folder)
