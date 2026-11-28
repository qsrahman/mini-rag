"""Per-project layout: a `.mini-rag/` directory marks a document folder, found like `.git`.

A project folder may come from someone else (a clone, a download), so its contents are not
trusted: only harmless settings are read from it, and symlinks inside it are refused.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from mini_rag.models import MiniRagError

logger = logging.getLogger(__name__)

PROJECT_DIR = ".mini-rag"

# Everything else (Ollama host, system prompt, paths, ...) can only come from the user's own
# environment or command line: from a project it could send documents elsewhere.
PROJECT_CONFIG_KEYS = frozenset(
    {
        "RAG_LLM_MODEL",
        "RAG_EMBEDDING_MODEL",
        "RAG_CHUNK_SIZE",
        "RAG_CHUNK_OVERLAP",
        "RAG_TOP_K",
        "RAG_RETRIEVAL",
        "RAG_RERANK",
        "RAG_THINK",
        "RAG_NUM_CTX",
    }
)
_MODEL_KEYS = frozenset({"RAG_LLM_MODEL", "RAG_EMBEDDING_MODEL"})
# Ollama runs these on its hosted service, even through the local server.
_CLOUD_MODEL = re.compile(r"[:-]cloud$", re.IGNORECASE)

_CONFIG_TEMPLATE = """\
# Settings for this project. Uncomment to change. Environment variables and command-line
# options take priority over this file. Only the settings listed here are read from it.
# RAG_LLM_MODEL=qwen3.5:9b-mlx
# RAG_EMBEDDING_MODEL=nomic-embed-text-v2-moe
# RAG_CHUNK_SIZE=400
# RAG_CHUNK_OVERLAP=60
# RAG_TOP_K=5
# RAG_RETRIEVAL=hybrid
# RAG_RERANK=false
# RAG_THINK=false
# RAG_NUM_CTX=8192
"""

_IGNORE_TEMPLATE = """\
# Files and folders that `mini-rag index` skips, one pattern per line (* and ? wildcards).
# A pattern without "/" matches a name anywhere; with "/" it matches a path from the project
# root. Hidden entries are always skipped. Delete a line to index those files again.
node_modules
venv
__pycache__
site-packages
build
dist
target
"""


class ProjectError(MiniRagError):
    pass


@dataclass(frozen=True)
class Project:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / PROJECT_DIR

    @property
    def vector_db_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def history_path(self) -> Path:
        return self.data_dir / "chat_history"

    def _read(self, name: str) -> str:
        """Text of a project file, or "" if there is none."""
        path = self.data_dir / name
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ProjectError(f"Cannot read {path}: {exc}") from exc

    def config(self) -> dict[str, str]:
        """Allowed settings from config.env; anything else is ignored with a warning."""
        path = self.data_dir / "config.env"
        text = io.StringIO(self._read(path.name))
        settings = {}
        for key, value in dotenv_values(stream=text, interpolate=False).items():
            if value is None:
                continue
            if key not in PROJECT_CONFIG_KEYS:
                logger.warning("%s: ignoring %s (not allowed in a project config)", path, key)
            elif key in _MODEL_KEYS and _CLOUD_MODEL.search(value.strip()):
                logger.warning("%s: ignoring cloud model %r for %s", path, value, key)
            else:
                settings[key] = value
        return settings

    def ignore_patterns(self) -> Sequence[str]:
        lines = (line.strip() for line in self._read("ignore").splitlines())
        return [line.rstrip("/") for line in lines if line and not line.startswith("#")]


def _checked(root: Path) -> Project:
    project = Project(root)
    entries = [project.data_dir, *project.data_dir.rglob("*")]
    for entry in entries:
        if entry.is_symlink():
            # A shipped symlink could make us write somewhere else (e.g. the chat history).
            raise ProjectError(f"Refusing to use project: {entry} is a symbolic link")
    return project


def open_project(explicit: Path | None = None, start: Path | None = None) -> Project:
    """The project at `explicit`, else the nearest one at or above `start` (default: cwd)."""
    if explicit is not None:
        root = explicit.resolve()
        if not (root / PROJECT_DIR).is_dir():
            raise ProjectError(f"No mini-rag project at {root}. Run: mini-rag init {explicit}")
        return _checked(root)
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / PROJECT_DIR).is_dir():
            return _checked(directory)
    raise ProjectError(f"No mini-rag project here ({here}). Run `mini-rag init` first.")


def init_project(directory: Path) -> tuple[Project, Path | None]:
    """Create a project. Returns it and the enclosing project's root, if there is one."""
    root = directory.resolve()
    if not root.is_dir():
        raise ProjectError(f"Not a directory: {directory}")
    if root == Path.home().resolve() or root == Path(root.anchor):
        raise ProjectError(f"Refusing to make {root} a project; choose a document folder")
    data_dir = root / PROJECT_DIR
    if data_dir.exists() or data_dir.is_symlink():
        raise ProjectError(f"{root} already has a {PROJECT_DIR} directory")
    outer = next((p for p in root.parents if (p / PROJECT_DIR).is_dir()), None)
    data_dir.mkdir()
    (data_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    (data_dir / "config.env").write_text(_CONFIG_TEMPLATE, encoding="utf-8")
    (data_dir / "ignore").write_text(_IGNORE_TEMPLATE, encoding="utf-8")
    return Project(root), outer
