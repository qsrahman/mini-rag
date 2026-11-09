"""Command-line entry point: argparse dispatch and service wiring."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

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
from mini_rag.indexer import Indexer
from mini_rag.interactive import ChatShell, format_answer, working
from mini_rag.models import MiniRagError
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
    common.add_argument("--db-path", type=Path, dest="vector_db_path", help="ChromaDB directory")
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

    parser = argparse.ArgumentParser(
        prog="mini-rag", description="Chat with your documents using local models."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", parents=[common], help="index a directory of documents")
    index.add_argument("directory", type=Path)
    index.add_argument("--rebuild", action="store_true", help="drop the collection and re-index")
    index.add_argument(
        "--prune",
        action="store_true",
        help="also remove indexed documents under DIRECTORY whose files no longer exist",
    )
    index.add_argument("--chunk-size", type=int, help="chunk size in (approximate) tokens")
    index.add_argument("--chunk-overlap", type=int, help="chunk overlap in (approximate) tokens")

    ask = commands.add_parser("ask", parents=[query], help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--source", help="only search the file with this source name")

    commands.add_parser("chat", parents=[query], help="interactive chat")

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
    overrides = {
        key: getattr(args, key, None)
        for key in (
            "llm_model",
            "embedding_model",
            "ollama_host",
            "vector_db_path",
            "collection_name",
            "top_k",
            "retrieval",
            "rerank",
            "chunk_size",
            "chunk_overlap",
        )
    }
    return Settings.from_env().with_overrides(**overrides)


def run_index(args: argparse.Namespace, settings: Settings) -> int:
    if not args.directory.is_dir():
        raise MiniRagError(f"Not a directory: {args.directory}")
    services = build_services(settings, rebuild=args.rebuild)
    indexer = Indexer(
        services.embedder, services.store, settings.chunk_size, settings.chunk_overlap
    )
    with working("Indexing documents..."):
        summary = indexer.index_directory(args.directory, prune=args.prune)
    print("Document indexing completed.\n")
    print(f"Files discovered: {summary.discovered}")
    print(f"Files indexed: {summary.indexed}")
    print(f"Files unchanged: {summary.unchanged}")
    print(f"Files failed: {len(summary.failed)}")
    print(f"Chunks created: {summary.chunks_created}")
    if args.prune:
        print(f"Documents pruned: {summary.pruned}")
        if summary.unprunable:
            print(
                f"note: {summary.unprunable} document(s) were indexed before paths were recorded "
                "and cannot be pruned; use --rebuild to purge them",
                file=sys.stderr,
            )
    for path, reason in summary.failed:
        print(f"  FAILED {path}: {reason}", file=sys.stderr)
    return 1 if summary.failed else 0


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


def run_eval(args: argparse.Namespace, settings: Settings) -> int:
    if args.k <= 0:
        raise MiniRagError(f"--k must be positive, got {args.k}")
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
    load_dotenv()
    handlers = {"index": run_index, "ask": run_ask, "chat": run_chat, "eval": run_eval}
    try:
        return handlers[args.command](args, _settings_from_args(args))
    except MiniRagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
