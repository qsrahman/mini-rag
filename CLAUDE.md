# CLAUDE.md

Mini RAG: a small, fully local RAG CLI (Ollama + ChromaDB + prompt_toolkit). See `README.md` for
usage and `PLAN.md` for design decisions and roadmap.

## Commands

Use `uv` only (no pip/Poetry/Conda). Python `>=3.14`.

```bash
uv sync
uv run pytest                  # no Ollama needed; `live` tests are deselected
uv run pytest -m live          # real-Ollama end-to-end tests (~30 s with the default model)
uv run ruff check . && uv run ruff format .
uv run mypy src/               # strict
uv run mini-rag --help
```

Add dependencies with `uv add` / `uv add --dev`; commit `uv.lock`.

CI (`.github/workflows/ci.yml`) runs the checks above except `live` on Linux and macOS with
`uv sync --locked`. Actions are pinned to commit hashes; update the version comment with the hash.

## Layout

`src/mini_rag/`: `cli.py` (argparse + wiring), `project.py` (`.mini-rag/` projects), `interactive.py` (prompt_toolkit shell and output
formatting), `config.py`, `models.py`, `indexer.py`, `retriever.py`, `generator.py`, `rag.py`,
`embeddings.py`, `evaluation.py`, `keyword_index.py`, `reranker.py`, `vector_store.py`, `ollama_utils.py`, `ingestion/{loaders,chunker}.py`.
Tests in `tests/unit` and `tests/integration`; shared fakes in `tests/conftest.py`.

## Architecture rules

- Dependencies point one way: `cli` → `interactive`/`rag`/`indexer` → `retriever`/`generator`/
  `embeddings`/`vector_store` → `models`. Keep it that way.
- `rag.py` does no parsing, chunking or Chroma calls. `retriever.py` does no generation.
  `interactive.py` is presentation only and calls `RagService`, the same as `ask`.
- `embeddings.py` is independent of the vector store. A new embedder only has to satisfy the
  `Embedder` protocol (`model` attribute plus `embed(texts)`).
- Chunk identity is `Chunk.id`. Settings are read from the environment through `_ENV_PARSERS`
  in `config.py`; a new one also needs the project allow-list and template if projects may set it,
  `.env.example` and the README table (`tests/unit/test_settings_consistency.py` checks them). A CLI
  option overrides the setting its `dest` is named after.
- `status` is read-only (`inspect_index`): it must not create, repair or reject an index.
- Expected failures subclass `MiniRagError` and are caught only in `cli.main` and the chat loop
  (`cli.main` also reports `OSError`). Translate third-party errors where they arise: Ollama via
  `ollama_utils`, SQLite in `KeywordIndex`, ChromaDB in `VectorStore`.
  Per-file `LoaderError`s are caught in `Indexer`; Ollama errors deliberately abort the run.
- No LangChain/LlamaIndex/Haystack. No external AI APIs. No `input()` for interaction.
- Retrieval changes should be justified with `mini-rag eval` on a frozen question set, not by intuition.
- No conversation memory or web UI (explicitly out of scope). Hybrid retrieval is the default;
  reranking is opt-in and off by default (see PLAN.md for the measurements).

## Invariants to preserve

- **Safe replacement:** `VectorStore.replace_document` upserts new chunks, then deletes stale ones.
  Chunk IDs are `{document_id}-{content_hash[:12]}-{chunk_index}`; embed before touching the store.
- **Fingerprint** = SHA-256 of file bytes + chunk size + overlap + `INDEX_VERSION`. Anything
  that changes chunk output must be part of it, or unchanged-file skipping will serve stale chunks.
  When loader or chunker output changes, increase `INDEX_VERSION` in `indexer.py` and update the
  pinned digest in `tests/integration/test_index_version.py` (the test fails until you do).
- **Embedding-model identity** is stored in collection metadata and checked on open; mismatch
  raises with a `--rebuild` instruction. Never hardcode embedding dimensions.
- **Sources come from retrieval, not the LLM.** `[S1]`... are assigned in `rag.build_sources`;
  unknown cited IDs are flagged, never trusted.
- **Prompt, context format and citation parsing are coupled:** `build_context` labels documents
  `<document id="S1">`, the system prompt tells the model to cite `[S1]`, and
  `find_unknown_citations` parses that style. Change them together.
- **ChromaDB is the source of truth; the FTS5 keyword index is derived.** Every write path in
  `VectorStore` (replace, delete) must update both; `_sync_keywords` repairs drift on open and
  runs on every command, so it compares ids only. Each keyword write is a single transaction
  (another process may open the index mid-write).
- **Readers follow other processes.** ChromaDB caches its vector index per process, so
  `VectorStore` watches the keyword database's `data_version` and closes and reopens the client when
  it changes; every Chroma write path must therefore end with a keyword-index commit. Refreshing
  must not write. Dense reads retry while a writer is mid-change. Test this with a real second
  process (`tests/integration/test_other_process.py`).
- **Check before destroying:** anything that can fail cheaply (ignore file, folder scan, Ollama
  probe) runs before `--rebuild` drops the collection. Never prune under a folder we could not list.
- **Parse failure and emptiness differ:** `LoaderError` keeps a document's old chunks,
  `EmptyDocumentError` removes them. `--rebuild` is only allowed for the whole project.
- **Everything is per project** (`.mini-rag/`). `source` and `document_id` come from the path
  relative to the project root; never store absolute paths, and never share a database between
  projects. Prune only touches documents under the indexed directory and never when no supported
  files are found there.
- **A project folder is untrusted input.** Only `PROJECT_CONFIG_KEYS` are read from `config.env`
  (nothing that can redirect data: host, prompt, paths, cloud models), symlinks inside `.mini-rag/`
  are refused, and no `.env` is loaded, including by ChromaDB (`_env_file=None`, Rust backend
  pinned). Think about this before adding a setting or a file the project can influence.
- The generator always sends `think` (default `False`): reasoning models otherwise think by default
  and answers take tens of seconds. Live Ctrl+C tests pass `--think` so generation is slow enough.
- Chroma metadata values are `str | int` only; omit `page` for non-PDF files rather than using None.
- Chunk sizes are character-based (~4 chars/token), documented in `chunker.py`. Never cut inside a
  word. `Settings.chunk_overlap` may be `None`; use `effective_chunk_overlap`.

## Testing conventions

- Tests must not need Ollama or model downloads. Fake the Ollama client (`MagicMock`) or use
  `FakeEmbedder` from `tests/conftest.py`; use `tmp_path` for ChromaDB.
- `ChatShell` takes an injectable `session` and `write` callback; test through those.
- `tests/e2e/test_live.py` (marker `live`) is the real end-to-end check; keep its LLM-text assertions
  loose. When typing into the pty, wait for the `You >` prompt before sending the next line.

## Style

Type hints everywhere (mypy strict). Ruff line length 100. Comments explain *why* only. Use
`logging` in internals; `print` belongs in `cli.py`/`interactive.py`. Don't suppress exceptions
silently. Check library signatures against the installed version before relying on them.
