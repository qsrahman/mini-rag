"""Each setting is listed in several places; these tests keep them in step."""

import re
from pathlib import Path

from mini_rag.cli import build_parser
from mini_rag.config import ENV_VARIABLES, Settings
from mini_rag.project import PROJECT_CONFIG_KEYS, init_project

ROOT = Path(__file__).resolve().parents[2]
# Set from the project, never from the environment.
FROM_PROJECT = {"project_root", "vector_db_path", "history_path"}


def names_in(text: str) -> set[str]:
    return set(re.findall(r"^#?\s*(RAG_\w+)=", text, re.MULTILINE))


def test_every_setting_can_come_from_the_environment():
    fields = set(Settings.__dataclass_fields__) - FROM_PROJECT
    assert {f"RAG_{name.upper()}" for name in fields} == ENV_VARIABLES


def test_project_keys_are_settings_and_match_the_template(tmp_path):
    assert PROJECT_CONFIG_KEYS <= ENV_VARIABLES
    init_project(tmp_path)
    assert names_in((tmp_path / ".mini-rag" / "config.env").read_text()) == PROJECT_CONFIG_KEYS


def test_env_example_lists_every_variable():
    listed = names_in((ROOT / ".env.example").read_text())
    assert listed == ENV_VARIABLES - {"RAG_SYSTEM_PROMPT"}  # a long text, documented in README


def test_readme_table_lists_every_variable():
    readme = (ROOT / "README.md").read_text()
    rows = re.findall(r"^\| `(RAG_\w+)` \|.*\| (yes|no) \|$", readme, re.MULTILINE)
    assert {name for name, _ in rows} == ENV_VARIABLES
    assert {name for name, allowed in rows if allowed == "yes"} == PROJECT_CONFIG_KEYS


def test_command_options_override_settings_by_name():
    args = build_parser().parse_args(
        ["ask", "q", "--top-k", "3", "--model", "m", "--collection", "c", "--no-think"]
    )
    named = {k: v for k, v in vars(args).items() if k in Settings.__dataclass_fields__}
    given = {k: v for k, v in named.items() if v is not None}
    assert given == {"top_k": 3, "llm_model": "m", "collection_name": "c", "think": False}
