"""Command-line entry point: argparse dispatch and service wiring."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mini_rag.config import Settings
from mini_rag.embeddings import OllamaEmbedder
from mini_rag.evaluation import (
    evaluate,
    format_report,
    generate_questions,
    load_questions,
    ollama_question_writer,
    write_questions,
)
from mini_rag.generator import Generator
from mini_rag.indexer import Indexer, scan
from mini_rag.interactive import ChatShell, format_answer, format_status, working
from mini_rag.models import MiniRagError
from mini_rag.project import Project, init_project, open_project
from mini_rag.rag import RagService
from mini_rag.reranker import LlmReranker
from mini_rag.retriever import Retriever
from mini_rag.vector_store import VectorStore


@dataclass
class Services:
    store: VectorStore
    embedder: OllamaEmbedder
    retriever: Retriever
    rag: RagService


def build_services(settings: Settings, rebuild: bool = False) -> Services:
    embedder = OllamaEmbedder(settings.embedding_model, settings.ollama_host)
    if rebuild:
        # Rebuilding drops the collection first; make sure Ollama works before destroying data.
        embedder.embed(["ping"])
    store = VectorStore(
        settings.vector_db_path,
        settings.collection_name,
        settings.embedding_model,
        rebuild=rebuild,
    )
    generator = Generator(
        settings.llm_model,
        settings.ollama_host,
        settings.system_prompt,
        settings.num_ctx,
        settings.think,
    )
    reranker = (
        LlmReranker(settings.llm_model, settings.ollama_host, min(settings.num_ctx, 4096))
        if settings.rerank
        else None
    )
    retriever = Retriever(embedder, store, settings.top_k, settings.retrieval, reranker)
    return Services(store, embedder, retriever, RagService(retriever, generator))


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--project", type=Path, help="project folder (default: found from the current folder)"
    )
    common.add_argument("--collection", dest="collection_name", help="collection name")
    common.add_argument("--embedding-model", help="Ollama embedding model")
    common.add_argument("--ollama-host", help="Ollama server URL")
    common.add_argument(
        "--retrieval", choices=["dense", "hybrid"], help="vectors only, or vectors + BM25 (RRF)"
    )
    common.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        help="rescore top candidates with the chat model (slower, usually better order)",
    )
    common.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")

    query = argparse.ArgumentParser(add_help=False, parents=[common])
    query.add_argument("--top-k", type=int, help="number of chunks to retrieve")
    query.add_argument("--model", dest="llm_model", help="Ollama chat model")
    query.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        help="let the model reason before answering (much slower)",
    )

    parser = argparse.ArgumentParser(
        prog="mini-rag", description="Chat with your documents using local models."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="make a folder a mini-rag project")
    init.add_argument("directory", type=Path, nargs="?", default=Path("."))
    init.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")

    index = commands.add_parser("index", parents=[common], help="index the project's documents")
    index.add_argument(
        "directory", type=Path, nargs="?", help="folder inside the project (default: all of it)"
    )
    index.add_argument(
        "--rebuild", action="store_true", help="drop the collection and re-index the whole project"
    )
    index.add_argument(
        "--prune",
        action="store_true",
        help="also remove indexed documents whose files were deleted or are now ignored",
    )
    index.add_argument("--chunk-size", type=int, help="chunk size in (approximate) tokens")
    index.add_argument("--chunk-overlap", type=int, help="chunk overlap in (approximate) tokens")

    ask = commands.add_parser("ask", parents=[query], help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--source", help="only search this file (path relative to the project)")

    commands.add_parser("chat", parents=[query], help="interactive chat")
    commands.add_parser("status", parents=[query], help="show the project and index in use")

    evaluation = commands.add_parser(
        "eval", parents=[common], help="measure retrieval quality on a frozen question set"
    )
    evaluation.add_argument("questions", type=Path, help="JSONL question file")
    evaluation.add_argument(
        "--generate", type=int, metavar="N", help="write N new questions from sampled chunks"
    )
    evaluation.add_argument("--model", dest="llm_model", help="Ollama model that writes questions")
    evaluation.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    evaluation.add_argument("--k", type=int, default=10, help="retrieval depth (default 10)")
    evaluation.add_argument(
        "--force", action="store_true", help="overwrite an existing question file"
    )
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    project = open_project(args.project)
    overrides = {
        key: getattr(args, key, None)
        for key in (
            "llm_model",
            "embedding_model",
            "ollama_host",
            "collection_name",
            "top_k",
            "retrieval",
            "rerank",
            "think",
            "chunk_size",
            "chunk_overlap",
        )
    }
    return Settings.for_project(project).with_overrides(**overrides)


def run_init(args: argparse.Namespace) -> int:
    project, outer = init_project(args.directory)
    print(f"Created project in {project.root}")
    if outer:
        print(f"note: this folder is inside another project: {outer}", file=sys.stderr)
    print("Next: run `mini-rag index` from this folder.")
    return 0


def _check_rebuildable(root: Path, ignore: Sequence[str]) -> None:
    found = scan(root, root, ignore)
    if found.unreadable:
        folders = ", ".join(str(folder) for folder, _ in found.unreadable)
        raise MiniRagError(
            f"Not rebuilding, the index is unchanged: cannot read {folders}. Fix the "
            "permissions or list the folder in .mini-rag/ignore."
        )
    if not found.files:
        raise MiniRagError(
            f"Not rebuilding, the index is unchanged: no supported files found under {root}. "
            "To empty the index, delete .mini-rag/chroma."
        )


def run_index(args: argparse.Namespace, settings: Settings) -> int:
    root = settings.project_root
    directory = root if args.directory is None else args.directory.resolve()
    # Check before building services: --rebuild must not drop data for a bad argument.
    if not directory.is_dir():
        raise MiniRagError(f"Not a directory: {args.directory}")
    if not directory.is_relative_to(root):
        raise MiniRagError(f"{directory} is outside the project {root}")
    if args.rebuild and directory != root:
        raise MiniRagError(
            "--rebuild drops the whole project index, so it cannot be limited to a folder. "
            "Run `mini-rag index --rebuild`, or leave out --rebuild to re-index just that folder."
        )
    # Read everything that can fail before --rebuild drops the existing index.
    ignore = Project(root).ignore_patterns()
    if args.rebuild:
        _check_rebuildable(root, ignore)
    services = build_services(settings, rebuild=args.rebuild)
    indexer = Indexer(
        services.embedder,
        services.store,
        settings.chunk_size,
        settings.chunk_overlap,
        root,
        ignore,
    )
    with working("Indexing documents..."):
        summary = indexer.index_directory(directory, prune=args.prune)
    print("Document indexing completed.\n")
    print(f"Files discovered: {summary.discovered}")
    print(f"Files indexed: {summary.indexed}")
    print(f"Files unchanged: {summary.unchanged}")
    print(f"Files failed: {len(summary.failed)}")
    print(f"Chunks created: {summary.chunks_created}")
    if args.prune:
        print(f"Documents pruned: {summary.pruned}")
    if summary.unreadable:
        print(f"Folders unreadable: {len(summary.unreadable)}")
    for path, reason in [*summary.unreadable, *summary.failed]:
        print(f"  FAILED {path}: {reason}", file=sys.stderr)
    return 1 if summary.failed or summary.unreadable else 0


def run_ask(args: argparse.Namespace, settings: Settings) -> int:
    services = build_services(settings)
    where = {"source": args.source} if args.source else None
    with working("Searching documents and generating an answer..."):
        answer = services.rag.ask(args.question, where=where)
    print(format_answer(answer))
    return 0


def run_chat(args: argparse.Namespace, settings: Settings) -> int:
    services = build_services(settings)
    return ChatShell(services.rag, settings, services.store.count).run()


def run_status(args: argparse.Namespace, settings: Settings) -> int:
    services = build_services(settings)
    print(format_status(settings, services.store.count()))
    return 0


def run_eval(args: argparse.Namespace, settings: Settings) -> int:
    if args.k <= 0:
        raise MiniRagError(f"--k must be positive, got {args.k}")
    if args.generate is not None and args.generate <= 0:
        raise MiniRagError(f"--generate must be positive, got {args.generate}")
    services = build_services(settings)
    if args.generate is None:
        questions = load_questions(args.questions)
        report = evaluate(
            questions,
            lambda q, k: services.retriever.retrieve(q, top_k=k),
            services.store.chunk_ids(),
            args.k,
        )
        print(format_report(report))
        return 0

    if args.questions.exists() and not args.force:
        # The set must stay frozen for before/after comparisons to mean anything.
        raise MiniRagError(f"{args.questions} already exists; use --force to overwrite it")
    samples = services.store.sample(args.generate, args.seed)
    if not samples:
        raise MiniRagError("No chunks long enough to sample; index some documents first")
    writer = ollama_question_writer(settings.llm_model, settings.ollama_host, settings.num_ctx)
    with working(f"Writing {len(samples)} questions with {settings.llm_model}..."):
        questions = generate_questions(samples, writer)
    write_questions(args.questions, questions)
    print(f"Wrote {len(questions)} questions to {args.questions}")
    print(f"Score them with: mini-rag eval {args.questions}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    handlers = {
        "index": run_index,
        "ask": run_ask,
        "chat": run_chat,
        "status": run_status,
        "eval": run_eval,
    }
    try:
        if args.command == "init":
            return run_init(args)
        return handlers[args.command](args, _settings_from_args(args))
    except (MiniRagError, OSError) as exc:  # OSError: unwritable folder, full disk, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
