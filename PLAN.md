# PLAN.md

## Goal

A small, reliable, understandable, fully local RAG application with a polished interactive CLI.
Not a framework. All processing, embedding and inference stay on the machine.

## Status

The foundation described below is implemented and verified: 264 unit/integration tests pass, ruff
and `mypy --strict` are clean.

`uv run pytest -m live` (9 tests, about 30 s) passed against local Ollama (`qwen3.5:9b-mlx`,
`nomic-embed-text-v2-moe`): indexing and grounded, cited answers for TXT, Markdown, DOCX and PDF
(with page numbers), refusal of an unanswerable question, no-op re-indexing, and Ctrl+C during a
running generation. In a real pty, chat prints "Cancelled." and stays usable, and one-shot `ask`
exits 130.

## Pipelines

- **Indexing:** discover files → load (PDF per page, DOCX, TXT/MD) → chunk → embed → replace the
  document's chunks in ChromaDB. Unchanged files are skipped.
- **Querying:** question → embed → cosine top-k search (optional metadata filter) → `[S1]..[Sn]`
  context → local LLM → answer plus the actually retrieved sources.

## Design decisions

| Decision | Reason |
|---|---|
| Package `mini_rag` (entry `mini_rag.cli:main`) | Matches the repo's existing uv package; spec named both `local_rag` and `mini_rag` |
| `indexer.py` separate from `rag.py` | RAG orchestration must not touch parsing, chunking or storage |
| `--rebuild` refused for a subfolder | It drops the whole collection, so it silently discarded every other folder |
| Files that now hold no text lose their chunks; unparseable ones keep them | Old answers from emptied files are wrong; a half-written or corrupt file should not destroy good content |
| Keyword writes are single transactions; open compares ids only and repairs duplicates | A concurrent open could duplicate rows permanently; loading all text on each command was wasteful |
| Reader reopens ChromaDB when `keywords.sqlite3` `data_version` changes | `chat` otherwise missed new chunks in vector search and failed after a rebuild; a new client without `close()` reuses the cached index |
| Dense reads retry, then drop vanished matches or report "being updated" | Reading mid-write raised errors or returned deleted chunks (this happened before the refresh too) |
| Model thinking off by default (`RAG_THINK`, `--think`) | About 19x faster on the default model with no clear accuracy difference; see findings below |
| Unreadable subfolders are skipped and reported; `--prune` leaves their documents | One folder must not stop a run, and "could not list" is not "deleted" |
| `--rebuild` scans first and refuses on unreadable folders or no files | A broken ignore file, an unreadable folder or an empty directory each wiped the index |
| Third-party errors become `MiniRagError`s at the module that calls them | Users get a message and a remedy instead of a traceback |
| CI on Linux and macOS with a locked sync; actions pinned to commits | Catches breakage on push without needing Ollama |
| Chunk ID embeds content hash | New chunks never overwrite old ones, so replacement is add-then-delete and failure-safe |
| Fingerprint includes chunk size/overlap | Changing chunking re-indexes without `--rebuild` |
| `INDEX_VERSION` in the fingerprint, pinned to a digest of loader and chunker output by a test | Code changes reach already indexed files; version 1 leaves existing fingerprints unchanged |
| `chromadb` capped below version 2 | `uv tool install` ignores the lockfile, and the app relies on Chroma internals (backend path, process cache) |
| Embedding model name stored on the collection | Prevents silently mixing incompatible vectors |
| Hybrid retrieval (dense + BM25, RRF k=60) is the default; `dense` stays available | Measured gain on the eval; exact identifiers are common in documents |
| BM25 via SQLite FTS5 in a derived `keywords.sqlite3`, repaired from ChromaDB on open | Stdlib, no new dependency, persistent; avoids a second source of truth |
| Per-project `.mini-rag/` found like `.git`; no global index | The command is installed globally; an index per document folder behaves the same from any subfolder |
| Document ids and sources are project-relative | Projects can be moved; equal file names in different folders stay distinct |
| Project config is allow-listed; symlinks refused; cloud models rejected; no `.env` loading | A downloaded folder must not redirect data or change behaviour |
| ChromaDB built with `_env_file=None` and the Rust backend pinned | It otherwise reads a `.env` from the current folder, including class paths to import |
| No database or history path settings | Relative ids make a shared database unsafe |
| Default ignore patterns are written into `.mini-rag/ignore` | Visible and editable instead of hidden behaviour |
| Reranking is opt-in, via the chat model's yes/no logprobs | No local rerank API; benefit was inconsistent, so default off |
| `mini-rag eval`: synthetic question per chunk, frozen JSONL, hit-rate/MRR/NDCG | Adapted from the hybrid-retrieval cookbook with a local LLM; lets retrieval changes be measured |
| Sources tracked outside the LLM | Citations can't be fabricated into the source list; unknown IDs produce a warning |
| Ollama errors abort indexing, loader errors don't | Ollama problems affect every file; a bad file affects only itself |
| Character-based chunk sizing (~4 chars/token) | Avoids a tokenizer dependency |
| Default chunk 400 / overlap 15% (60) | Keeps chunks under the embedding model's 512-token input limit; an unset overlap follows the chunk size |
| Oversized text splits at sentences, lines, then words (index version 2) | The old character cut split words and lines; see findings below |
| One-off chunk flags print a note | They are not persisted; `config.env` is where lasting values belong |
| Each file is read once; loaders parse bytes | The indexer hashes and parses the same bytes, so there is no double read and no hash/parse mismatch if the file changes mid-run |
| `os.walk` with directory pruning for discovery | Hidden directories (`.git`, ...) are never descended into |
| `index --rebuild` probes the embedder before dropping the collection | An unreachable Ollama can't wipe the existing index |
| `RAG_NUM_CTX` (default 8192) | Ollama's default context could silently truncate retrieved chunks |
| System prompt: grounded-only, fixed decline sentence, partial answers, conflict flagging, document text is data | Researched practice (Anthropic hallucination and long-context guidance, indirect-injection literature); see README "System prompt" |
| Context as `<document id="S1">` XML blocks, documents first, question last | Anthropic's long-context structure; IDs match the `[S1]` citations the app verifies |
| Chroma telemetry disabled | "Fully local" |
| `Embedder` protocol, concrete classes elsewhere | Only the embedder has a real second-implementation need |

## Known issues and limitations

1. **Chunk size is estimated, not tokenized.** The default of 400 tokens (~1600 characters) fits
   `nomic-embed-text-v2-moe`'s 512-token limit for English prose, but non-English or code-heavy text
   can tokenize denser and still exceed it.
2. Moving a file inside a project re-indexes it; the old entry stays until `index --prune`.
3. A project's `ignore` file and any prebuilt index are trusted as data; users should create their
   own with `init`. Indexes from the old `./storage` layout are not migrated.
4. Scanned PDFs have no OCR and are reported as failed.
5. If another process writes continuously with no pauses, each search can take about 1 s of retries.
   With pauses of 50 ms or more between documents, searches stayed at about 3 ms.
6. Live tests assert LLM text loosely (model output is nondeterministic), so a regression in answer
   quality would not necessarily fail them.

## Retrieval findings (from the ai-cookbook review)

Reviewed `knowledge/hybrid-retrieval` and `knowledge/agentic-rag`. Outcome:

- **Eval harness: done** (`mini-rag eval`), the cookbook's central lesson.
- **Nomic task prefixes: tried and reverted, no measured gain.** The model card expects
  `search_query:` / `search_document:` prefixes, but measuring showed no benefit (and a forced
  rebuild plus extra protocol surface), so the code was removed. Same chunks, same frozen
  questions, through Ollama:

  | Corpus | Questions | Setup | hit@1 | hit@5 | hit@10 | MRR | NDCG@10 |
  |---|---|---|---|---|---|---|---|
  | 72 chunks (this repo's docs + a prompting guide) | 40 | no prefix | 57.5% | 92.5% | 92.5% | 0.717 | 0.769 |
  | | | with prefixes | 57.5% | 87.5% | 92.5% | 0.692 | 0.748 |
  | 723 chunks (package READMEs) | 120 | no prefix | 48.3% | 80.8% | 88.3% | 0.626 | 0.689 |
  | | | with prefixes | 45.0% | 80.8% | 88.3% | 0.612 | 0.679 |

  Differences are within noise at these sample sizes. Synthetic questions share vocabulary with
  their chunk, which may mask the benefit; revisit if a future eval corpus is less lexical. Baseline for future work: 723-chunk corpus,
  hit@5 80.8%, MRR about 0.62.
- **Hybrid BM25 + RRF: implemented, default.** BM25 uses SQLite FTS5 (stdlib). Chroma's own
  `Search`/`Rrf` API was passed over: it needs a sparse-vector schema and sparse embeddings stored at
  write time. Same frozen questions and chunks, k=10:

  | Corpus | Mode | hit@1 | hit@5 | hit@10 | MRR | NDCG@10 |
  |---|---|---|---|---|---|---|
  | 723 chunks, 120 questions | dense | 48.3% | 81.7% | 89.2% | 0.628 | 0.692 |
  | | hybrid | 54.2% | 83.3% | 90.8% | 0.670 | 0.728 |
  | 72 chunks, 40 questions | dense | 57.5% | 92.5% | 92.5% | 0.717 | 0.769 |
  | | hybrid | 75.0% | 95.0% | 100.0% | 0.838 | 0.877 |

  Caveat: generated questions share wording with their chunk, which favours keyword matching, so
  real-world gains may be smaller.
- **Stopword removal in keyword search: implemented** (idea from reviewing ARahim3/DigUp). English
  function words are dropped from the BM25 query unless nothing else is left; no re-indexing
  needed. Measured with `mini-rag eval` (hybrid, k=10) on newer question sets, so not comparable
  with the tables above:

  | Corpus | Keyword query | hit@1 | hit@5 | hit@10 | MRR | NDCG@10 |
  |---|---|---|---|---|---|---|
  | 728 chunks, 120 questions | all words | 50.8% | 80.8% | 87.5% | 0.640 | 0.698 |
  | | stopwords removed | 55.0% | 84.2% | 88.3% | 0.671 | 0.723 |
  | 40 chunks, 40 questions | all words | 70.0% | 92.5% | 92.5% | 0.775 | 0.812 |
  | | stopwords removed | 70.0% | 92.5% | 97.5% | 0.790 | 0.835 |

  The list is English only; other languages get no filtering.
- **Other DigUp ideas.** EmbeddingGemma 2 with task prompts scored higher pooled over both sets
  but less consistently (needs prompt support in the embedder and `--rebuild`); not adopted yet.
  Score-based fusion instead of RRF, a relevance cutoff and file names in the search text showed no
  consistent gain. Untested: re-pointing moved files without re-embedding, size/mtime change
  detection.
- **Reranker: built as opt-in (`--rerank`), off by default, evidence mixed.** The cookbook uses
  Cohere's API (not local). Ollama 0.40.15 has no rerank endpoint (`/api/rerank` returns 404), and
  a local cross-encoder would add a heavy ML dependency, so candidates are scored by the chat model
  (yes/no probability from logprobs), top 10 candidates, about 1 s per candidate:

  | Corpus | Questions | Setup | hit@1 | hit@5 | MRR |
  |---|---|---|---|---|---|
  | 723 chunks | first 60 | hybrid | 55.0% | 80.0% | 0.656 |
  | | | hybrid + rerank | 61.7% | 86.7% | 0.731 |
  | 72 chunks | 40 | hybrid | 75.0% | 95.0% | 0.838 |
  | | | hybrid + rerank | 70.0% | 95.0% | 0.783 |

  It helps when hybrid leaves headroom and hurts when hybrid is already strong, for about 10 s of
  extra latency per question. The same model wrote the questions and scores the candidates, which may
  flatter it. Not enabled by default. Decision: kept as an opt-in feature.
- **Agentic loop: not pursued.** The cookbook's own comparison puts it at roughly 3-10x tokens;
  a poor fit for a local 9B thinking model and for PDF/DOCX sources.

## Thinking vs no thinking

`qwen3.5:9b-mlx`, 13 hand-written cases with fixed context (no retrieval), one run each; failed
cases were then read by hand:

| Setting | Correct | Median | Slowest |
|---|---|---|---|
| thinking off | 12-13 of 13 | 2.0 s | 5.7 s |
| thinking on | 12 of 13 | 37.8 s | 137 s (292 s on a rerun) |

The weak case was date arithmetic ("90 days before 31 December"). With thinking on the model refused
to calculate (2 of 2 runs). With it off, three reruns each gave a different wrong date; the first run
passed the automatic check. So off fails confidently and on fails cautiously, at far higher cost. The
automatic checks mis-scored two correct answers, corrected by hand. A small, single-model sample.

## Chunking change (index version 2)

Old versus new chunker, 400/60, hybrid retrieval:

| Measure | Old | New |
|---|---|---|
| Words split, generated lists (2,640 lines) | 178 | 0 |
| Words split, 74 package READMEs | 93 | 0 |
| Lines intact in some chunk (lists) | 2524/2640 | 2640/2640 |
| Complete requested line in top 5 (150 lookups) | 86 | 98 |
| Same, top 1 | 55 | 62 |
| READMEs, right document hit@5 (100 questions) | 96% | 94% |
| READMEs, document MRR | 0.927 | 0.921 |

A clear gain for lists; no measurable change on prose-heavy READMEs (the difference is two
questions). A 200,000-line file with no sentence breaks takes about 0.8 s to chunk.

## Roadmap (not started; each is optional and out of scope for v1)

1. Re-point moved files without re-embedding them.
2. A relevance-distance cutoff so weak matches aren't sent to the LLM (`RetrievedChunk.distance` is carried, but None for keyword-only hits).
3. Streaming answers in chat.
4. Conversation memory. A faster reranker (a real cross-encoder) only if the eval shows a clear need.

## Working agreements

- Keep changes small and traceable to a request; follow the rules in `CLAUDE.md`.
- Before declaring a change done: `uv run pytest`, `uv run ruff check .`, `uv run mypy src/`.
- Don't claim end-to-end success unless Ollama was actually exercised.
