# Mini RAG

Chat with your own documents using locally running models. Documents (PDF, DOCX, TXT, Markdown)
are chunked, embedded with Ollama, stored in a persistent ChromaDB collection, and retrieved to
ground answers from a local LLM. Nothing leaves your machine: no external AI APIs, no telemetry
(ChromaDB's is disabled).

No RAG framework is used; the pipeline is implemented directly on `ollama`, `chromadb`,
`pymupdf`, `python-docx` and `prompt_toolkit`.

## Architecture

```mermaid
flowchart LR
    subgraph Indexing
        D[Documents] --> L[Loaders] --> C[Chunker] --> E1[Embedder] --> V[(ChromaDB)]
    end
    subgraph Querying
        Q[Question] --> R[Retriever: vectors + BM25, RRF] --> X[Context builder] --> G[Local LLM] --> A[Answer + sources]
    end
    V --> R
```

**Indexing:** Document → Loader → Chunker → Embeddings → Vector Database
**Querying:** Question → Hybrid Retrieval (query embedding + BM25, fused by RRF) → Context Construction → Local LLM → Answer

| Module | Responsibility |
|---|---|
| `project.py` | `.mini-rag/` discovery and creation, filtered project config, ignore patterns |
| `config.py` | Defaults, `RAG_*` settings, validation |
| `models.py` | `Document`, `Chunk`, `RetrievedChunk`, `Source`, `Answer`, base error |
| `ingestion/loaders.py` | File discovery and text extraction (PDF pages keep page numbers) |
| `ingestion/chunker.py` | Paragraph/sentence-aware chunking with overlap |
| `embeddings.py` | `Embedder` protocol and the Ollama implementation (batched) |
| `evaluation.py` | Synthetic-question retrieval eval: question generation, hit-rate / MRR / NDCG |
| `vector_store.py` | ChromaDB persistence, cosine search, safe document replacement, model-compatibility check |
| `indexer.py` | Incremental indexing pipeline (load → chunk → embed → store) |
| `retriever.py` | Dense vector search, optionally fused with BM25 keyword search by Reciprocal Rank Fusion. No generation |
| `reranker.py` | Optional: rescores the top 10 candidates with the chat model (yes/no logprobs) |
| `keyword_index.py` | SQLite FTS5 (stdlib) BM25 index over chunk text, kept in step with ChromaDB by `vector_store.py` |
| `generator.py` | Ollama chat call with the system prompt |
| `rag.py` | Orchestration: retrieve, build `[S1]`-labelled context, generate, attribute sources |
| `interactive.py` | prompt_toolkit shell, answer formatting, spinner (presentation only) |
| `cli.py` | argparse dispatch and wiring |

Key behaviours:

- **Incremental indexing.** Each file gets a fingerprint (SHA-256 of its bytes plus the chunk
  size/overlap and an index version, which is raised when a new mini-rag release extracts or chunks
  text differently, so the next `index` re-processes your files). Unchanged files are skipped; changed files are re-embedded and replaced. Chunk IDs
  contain the fingerprint, so replacement first *adds* the new chunks and only then deletes the
  old ones. If embedding fails, the previous version stays intact.
- **Embedding-model safety.** The embedding model name is stored on the collection. Using a
  different model without `--rebuild` fails with an explanation instead of mixing vectors.
- **Honest citations.** The sources shown to you are the chunks that were actually retrieved
  (`[S1]`, `[S2]`, ...), tracked outside the LLM. If the model cites an ID that was not retrieved,
  a warning is printed. If nothing is retrieved (empty index), the LLM is not called.
- **Chunk size is approximate.** Sizes are counted in characters at ~4 characters per token to
  avoid a tokenizer dependency. Real token counts vary with language and content.

## Prerequisites

- macOS/Linux with Python 3.14+ (uv will fetch one if needed)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)

### Install uv

```bash
brew install uv          # or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Ollama and models

Install from <https://ollama.com/download> (or `brew install ollama`), start it
(`ollama serve`, or the desktop app), then:

```bash
ollama pull qwen3.5:9b-mlx
ollama pull nomic-embed-text-v2-moe
```

## Install

As a global command:

```bash
uv tool install git+https://github.com/qsrahman/mini-rag
```

From a local clone, use `uv tool install .` (add `--editable` while developing, or `--reinstall`
to pick up changes). `uv tool install` resolves dependencies fresh and does not use `uv.lock`.

For development, work in the clone with `uv sync` and prefix commands with `uv run`.

## Quick start

```bash
cd ~/work/contracts
```

```bash
mini-rag init
```

```bash
mini-rag index
```

```bash
mini-rag ask "What is the notice period?"
```

## Projects

`mini-rag init [folder]` turns a document folder into a project by creating `.mini-rag/`:

```text
contracts/
├── .mini-rag/
│   ├── config.env     settings for this project
│   ├── ignore         files and folders `index` skips
│   ├── chroma/        vectors and keywords.sqlite3
│   ├── chat_history
│   └── .gitignore     contains "*", so none of this is committed
└── leases/office.pdf
```

- Every other command uses the nearest `.mini-rag/` in the current folder or its parents, so it
  works from any subfolder. `--project PATH` picks one explicitly (no parent search).
- Nothing is created implicitly. `init` refuses your home folder and `/`.
- Documents are identified by their path relative to the project, so you can move or rename the
  project folder without re-indexing.
- Each project has its own index; one question searches one project. To remove a project, delete
  its `.mini-rag/` folder.
- `ignore` takes one pattern per line (`*`, `?`). A pattern without `/` matches a name anywhere;
  with `/` it matches a path from the project root. It starts with `node_modules`, `venv`,
  `__pycache__`, `site-packages`, `build`, `dist` and `target`; delete lines you want indexed.
  Hidden files and folders are always skipped.

### Folders from other people

A `.mini-rag/` folder may arrive with a clone or download, so it is not trusted:

- Only the settings marked "yes" under Configuration are read from `config.env`. Others (such as the
  Ollama host or system prompt) are ignored with a warning, as are Ollama cloud models
  (`...:cloud`), which would send your documents off the machine.
- Symbolic links inside `.mini-rag/` are refused.
- `.env` files are not loaded.
- A ready-made index can contain any text, and the `ignore` file decides what gets indexed. Do not
  rely on someone else's `.mini-rag/`: delete it and run `mini-rag init` yourself.

## Configuration

Priority: defaults, then the project's `.mini-rag/config.env`, then environment variables, then
command-line options. [.env.example](.env.example) lists the variables.

| Variable | Default | CLI option | In `config.env` |
|---|---|---|---|
| `RAG_LLM_MODEL` | `qwen3.5:9b-mlx` | `--model` (ask, chat) | yes |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text-v2-moe` | `--embedding-model` | yes |
| `RAG_CHUNK_SIZE` | `400` | `--chunk-size` (index) | yes |
| `RAG_CHUNK_OVERLAP` | `60` | `--chunk-overlap` (index) | yes |
| `RAG_TOP_K` | `5` | `--top-k` (ask, chat) | yes |
| `RAG_RETRIEVAL` | `hybrid` | `--retrieval dense\|hybrid` | yes |
| `RAG_RERANK` | `false` | `--rerank` / `--no-rerank` | yes |
| `RAG_THINK` | `false` | `--think` / `--no-think` (ask, chat) | yes |
| `RAG_NUM_CTX` | `8192` | | yes |
| `RAG_OLLAMA_HOST` | `http://localhost:11434` | `--ollama-host` | no |
| `RAG_COLLECTION_NAME` | `documents` | `--collection` | no |
| `RAG_SYSTEM_PROMPT` | built-in grounded-QA prompt | | no |

The database and chat history always live in the project's `.mini-rag/`.

Invalid values (non-positive sizes, overlap ≥ size, non-integers) are rejected at startup.

`RAG_NUM_CTX` is passed to Ollama as the context window for generation. Ollama's own default is
small enough that five 400-token chunks plus the question could be silently truncated.

> **Note on chunk size:** `nomic-embed-text-v2-moe` accepts at most 512 tokens, so the default of
> 400 (approximate) tokens leaves headroom. Sizes are estimated at ~4 characters per token; for
> non-English or code-heavy text the real token count can be higher, so lower `RAG_CHUNK_SIZE` if
> you see weak retrieval. Changing chunk settings re-indexes files automatically.

### System prompt

The default prompt lives in `config.py` (`DEFAULT_SYSTEM_PROMPT`); override it with
`RAG_SYSTEM_PROMPT`. It is built on common guidance for grounded RAG:

- Answer only from the supplied documents, and give the model explicit permission (and a fixed
  sentence) to decline when the answer is absent ([Anthropic: reduce hallucinations][halluc]).
- Cite a source ID after each claim, and never invent IDs, pages or quotes. The app independently
  checks citations against what was actually retrieved.
- Answer the supported part of a partly answerable question and say what is missing; flag
  contradictions between sources.
- Treat document text as data, not instructions, to blunt indirect prompt injection from
  indexed files. This helps but does not fully prevent it
  ([Rag and Roll, arXiv 2408.05025][ragroll]).
- The user message puts the documents first as `<document id="S1"><source>...</source>
  <document_content>...</document_content></document>` blocks and the question last
  ([Anthropic: long-context tips][longctx]).

If you replace the prompt, keep it consistent with that context format and with the `[S1]`
citation style, which `rag.py` parses to detect citations that match no retrieved source.

[halluc]: https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations
[longctx]: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#long-context-prompting
[ragroll]: https://arxiv.org/pdf/2408.05025

## Usage

### Retrieval modes

`hybrid` (default) runs dense vector search and BM25 keyword search, then merges the two ranked
lists with Reciprocal Rank Fusion (k=60). Common English function words ("what", "is", "the") are
ignored by the keyword search unless the query has nothing else. It takes `max(4 * top_k, 20)`
candidates from each. It helps with exact terms (identifiers, error codes, names, numbers) that
embeddings blur. `dense` is vectors only. The keyword index is a SQLite FTS5 file
(`keywords.sqlite3`, next to the ChromaDB data) that is derived from ChromaDB: if the two ever
disagree (for example on the first run after upgrading), it is rebuilt automatically. With
`--source`, hybrid search supports string equality filters on `source` only. Measured effect: see
PLAN.md.

### Optional reranking

`--rerank` (or `RAG_RERANK=true`) rescores the top 10 retrieved chunks with the chat model before
taking the top `k`. Ollama has no rerank API, so each candidate costs one LLM call (about a second
each, roughly 10 s extra per question). It is **off by default** because the measured benefit was
inconsistent: it helped on a larger, harder corpus and hurt on a small one where hybrid was already
strong (details in PLAN.md). Use `mini-rag eval ... --rerank` on your own documents before enabling it.

### Index documents

```bash
mini-rag index                 # the whole project
mini-rag index leases          # one folder inside the project
mini-rag index --rebuild       # drop the collection and re-index
mini-rag index --prune         # also drop documents that were deleted or are now ignored
mini-rag status                # project, models and number of indexed chunks
```

`--rebuild` drops the project's whole collection before indexing, so it is refused together with a
folder inside the project. It is also refused, leaving the index unchanged, if any folder cannot be
read or no supported files are found. Individual files that cannot be read or parsed during a rebuild
are absent from the new index.

A subfolder that cannot be read is reported under `Folders unreadable` and the rest is indexed;
`--prune` keeps what was previously indexed under it.

Discovers `.pdf`, `.docx`, `.txt`, `.md`/`.markdown` files recursively (hidden files and Word lock
files are skipped), then prints a summary. Corrupted or empty files, and PDFs with no text layer
(scanned; no OCR), are reported under `Files failed` without stopping the run; the exit code is 1 if
any file failed. A file that can no longer be parsed keeps its previously indexed content; one that
now holds no text at all has that content removed. A problem with Ollama itself (not running, model missing) aborts the run with
instructions.

### Thinking

Models that can "think" reason at length before answering. mini-rag turns this **off** by default:
on `qwen3.5:9b-mlx`, 13 test questions took a median of 2 s without thinking and 38 s with it, with
no clear difference in accuracy. Turn it on per question with `--think`. Neither mode was reliable
at date arithmetic, so check calculated answers against the cited source. Without thinking the model
tended to give a wrong date; with it, to say the date was not stated. Some models (for example
`gpt-oss`) still reason a little when thinking is off; a model that cannot think returns an error if
you pass `--think`.

### Ask one question

```bash
mini-rag ask "Explain supervised learning."
mini-rag ask "Explain neural networks." --top-k 8
mini-rag ask "What does chapter 2 say?" --source books/guide.pdf   # path relative to the project
```

### Measure retrieval quality

```bash
mini-rag eval questions.jsonl --generate 100   # once: a local LLM writes one question per sampled chunk
mini-rag eval questions.jsonl                  # score the current retriever (k=10 by default)
```

Each generated question is paired with the chunk it came from, which counts as the single correct
result. The report gives chunk hit@1/3/5/k, MRR, NDCG, and whether the right *document* was found.
Keep the question file frozen (generation refuses to overwrite without `--force`) and re-score
after each retrieval change; the numbers are only meaningful relative to each other. Questions are
tied to chunk IDs, so regenerate them if you change chunk size or the documents change. Generated
questions tend to share vocabulary with their chunk, which flatters keyword-style matching.
Recipe adapted from the [hybrid-retrieval cookbook](https://github.com/daveebbelaar/ai-cookbook/blob/main/knowledge/hybrid-retrieval/docs/build-your-own-eval.md).

### Interactive chat

```bash
mini-rag chat
```

Each question is answered independently (there is no conversation memory). You can run
`mini-rag index` in another terminal while a chat is open: the next question uses the updated index.
During a `--rebuild`, questions search whatever has been re-indexed so far, or ask you to try again.

| Command | Description |
|---|---|
| `/help` | Show commands and keys |
| `/status` | Project, model, database and index status |
| `/clear` | Clear the screen |
| `/exit`, `/quit` | Leave (also Ctrl+D) |

| Key | Action |
|---|---|
| Enter | Send |
| Alt+Enter / Ctrl+J | Newline (pasted multi-line text is kept) |
| Up / Down, Ctrl+R | History browse / search (persisted in `.mini-rag/chat_history`) |
| Ctrl+C | Cancel the current input, or a running answer |
| Ctrl+D | Exit |

## Project structure

```text
src/mini_rag/
  cli.py  interactive.py  project.py  config.py  models.py
  indexer.py  retriever.py  reranker.py  generator.py  rag.py
  embeddings.py  vector_store.py  keyword_index.py  evaluation.py  ollama_utils.py
  ingestion/  loaders.py  chunker.py
tests/
  unit/  integration/  e2e/  conftest.py
```

## Development

```bash
uv run pytest                 # no Ollama required (live tests are deselected)
uv run pytest -m live         # live end-to-end tests; needs Ollama + the default models (~30 s with the default model)
uv run pytest --cov=mini_rag
uv run ruff check .
uv run ruff format .
uv run mypy src/
```

GitHub Actions ([ci.yml](.github/workflows/ci.yml)) runs lint, format check, type check and the
non-live tests on Linux and macOS for pushes to `main` and for pull requests, and fails if
`uv.lock` is out of date. Live tests need Ollama, so run them locally.

`tests/e2e` runs the real pipeline against Ollama (all four formats, citations, refusal, and
Ctrl+C during generation in a pty); it skips itself if Ollama or the models are missing.
Other tests mock Ollama; integration tests use temporary ChromaDB directories and a deterministic fake
embedder.

### Dependency management

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # development dependency
uv sync                   # install from uv.lock
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to Ollama` | Start Ollama (`ollama serve`) or fix `RAG_OLLAMA_HOST` |
| `model ... is not available` | `ollama pull <model>` |
| `was built with embedding model ...` | You changed the embedding model; re-run `index --rebuild` |
| `no extractable text` for a PDF | It is likely scanned images; OCR is not supported |
| `No mini-rag project here` | Run `mini-rag init` in your document folder, or pass `--project` |
| Answers say nothing was found | Run `index` first, check `status` shows chunks, try a larger `--top-k` |
| Slow first answer | The model is loading into memory |
| Every answer is slow | Check `status`: with thinking on, answers take much longer |
| `does not support thinking` | Drop `--think` / `RAG_THINK` for this model |
| `Invalid Ollama host` | Use a URL such as `http://localhost:11434` |
| `Keyword index error ...` | If damaged, delete `.mini-rag/chroma/keywords.sqlite3`; it is rebuilt automatically |
| `Vector database error ...` | If damaged, delete `.mini-rag/chroma` and run `mini-rag index` |
| `Not rebuilding, the index is unchanged` | Fix the named folder's permissions, add it to `.mini-rag/ignore`, or check the files are there |

## Known limitations

- Moving or renaming a file *inside* a project indexes it as new; `index --prune` removes the old
  entry.
- `--prune` refuses to run if it finds no supported files where indexed documents live, so an empty
  or unmounted folder cannot wipe the index.
- Indexes created before projects existed (`./storage/chroma`) are not used; run `init` and `index`.
- There is no multi-turn memory, by design.
