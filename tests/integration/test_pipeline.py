"""Index real files into a temporary ChromaDB; Ollama is replaced by fakes."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mini_rag.cli import build_parser, main
from mini_rag.indexer import Indexer
from mini_rag.models import MiniRagError
from mini_rag.ollama_utils import OllamaError
from mini_rag.project import init_project
from mini_rag.rag import RagService
from mini_rag.retriever import Retriever
from mini_rag.vector_store import VectorStore
from tests.conftest import FakeEmbedder, make_docx, make_pdf


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "cooking.txt").write_text("Pasta needs boiling water and plenty of salt for flavour.")
    (root / "space.md").write_text("# Space\n\nRockets burn fuel to reach orbit around the planet.")
    make_pdf(root / "birds.pdf", ["Penguins are flightless birds living in cold climates."])
    make_docx(root / "money.docx", ["Compound interest grows savings over many years."])
    (root / "broken.pdf").write_bytes(b"garbage")
    (root / "empty.txt").write_text("")
    return root


@pytest.fixture
def store(tmp_path, embedder):
    return VectorStore(tmp_path / "db", "docs", embedder.model)


def make_indexer(embedder, store, root, size=200, overlap=20, ignore=()):
    return Indexer(embedder, store, root, size, overlap, ignore)


@pytest.fixture
def project(docs, monkeypatch):
    """`docs` as a project, with Ollama's embedder replaced; returns common CLI arguments."""
    init_project(docs)
    monkeypatch.setattr("mini_rag.cli.OllamaEmbedder", lambda model, host: FakeEmbedder(model))
    return ["--project", str(docs), "--embedding-model", "fake-embed"]


def project_store(docs):
    return VectorStore(docs / ".mini-rag" / "chroma", "documents", "fake-embed")


def test_query_commands_share_model_options():
    parser = build_parser()
    ask = parser.parse_args(["ask", "question", "--top-k", "7", "--model", "model-a"])
    chat = parser.parse_args(["chat", "--top-k", "9", "--model", "model-b"])

    assert (ask.top_k, ask.llm_model) == (7, "model-a")
    assert (chat.top_k, chat.llm_model) == (9, "model-b")


def test_index_summary_and_failures(docs, embedder, store):
    summary = make_indexer(embedder, store, docs).index_directory(docs)
    assert summary.discovered == 6
    assert summary.indexed == 4
    assert summary.unchanged == 0
    assert sorted(p.name for p, _ in summary.failed) == ["broken.pdf", "empty.txt"]
    assert summary.chunks_created == store.count() == 4


def test_second_run_is_a_no_op(docs, embedder, store):
    indexer = make_indexer(embedder, store, docs)
    indexer.index_directory(docs)
    embedder.calls.clear()
    summary = indexer.index_directory(docs)
    assert (summary.indexed, summary.unchanged) == (0, 4)
    assert embedder.calls == []
    assert store.count() == 4


def test_modified_file_is_reindexed_without_duplicates(docs, embedder, store):
    indexer = make_indexer(embedder, store, docs)
    indexer.index_directory(docs)
    (docs / "cooking.txt").write_text("Rice should be rinsed before steaming in a covered pot.")
    summary = indexer.index_directory(docs)
    assert (summary.indexed, summary.unchanged) == (1, 3)
    assert store.count() == 4
    hits = Retriever(embedder, store).retrieve("rinse rice steaming", top_k=10)
    texts = [h.chunk.text for h in hits]
    assert any("Rice" in t for t in texts)
    assert not any("Pasta" in t for t in texts)


def test_failed_embedding_keeps_old_content(docs, embedder, store):
    indexer = make_indexer(embedder, store, docs)
    indexer.index_directory(docs)
    (docs / "cooking.txt").write_text("Brand new text about bread.")
    embedder.embed = MagicMock(side_effect=RuntimeError("embedder down"))
    with pytest.raises(RuntimeError):
        indexer.index_directory(docs)
    assert store.count() == 4
    hits = Retriever(FakeEmbedder(), store).retrieve("pasta", top_k=10)
    assert any("Pasta" in h.chunk.text for h in hits)


def test_chunking_settings_change_triggers_reindex(docs, embedder, store):
    make_indexer(embedder, store, docs, size=200).index_directory(docs)
    summary = make_indexer(embedder, store, docs, size=100, overlap=10).index_directory(docs)
    assert summary.indexed == 4


def test_retrieval_and_answer_with_sources(docs, embedder, store):
    make_indexer(embedder, store, docs).index_directory(docs)
    generator = MagicMock()
    generator.generate.return_value = "Penguins cannot fly [S1]."
    rag = RagService(Retriever(embedder, store, top_k=2), generator)

    answer = rag.ask("Which birds are flightless penguins?")

    assert answer.sources[0].name == "birds.pdf"
    assert answer.sources[0].page == 1
    assert len(answer.sources) == 2
    _, context = generator.generate.call_args.args
    assert "Penguins are flightless" in context


def test_source_filter(docs, embedder, store):
    make_indexer(embedder, store, docs).index_directory(docs)
    hits = Retriever(embedder, store).retrieve("water", top_k=5, where={"source": "space.md"})
    assert {h.chunk.metadata["source"] for h in hits} == {"space.md"}


def test_cli_index_and_help(docs, project, capsys):
    code = main(["index", *project])
    out = capsys.readouterr()
    assert code == 1  # two files failed
    assert "Files discovered: 6" in out.out and "Files indexed: 4" in out.out
    assert "Files failed: 2" in out.out
    assert "broken.pdf" in out.err

    code = main(["index", *project])
    assert "Files unchanged: 4" in capsys.readouterr().out

    # Changing the embedding model without --rebuild is refused with instructions.
    code = main(["index", *project, "--embedding-model", "other"])
    assert code == 1 and "--rebuild" in capsys.readouterr().err

    code = main(["index", *project, "--embedding-model", "other", "--rebuild"])
    assert "Files indexed: 4" in capsys.readouterr().out


def test_cli_rejects_missing_directory(docs, project, tmp_path, capsys):
    assert main(["index", str(tmp_path / "nope"), *project]) == 1
    assert "Not a directory" in capsys.readouterr().err


def test_cli_refuses_to_index_outside_the_project(docs, project, tmp_path, capsys):
    main(["index", *project])
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    # Even with --rebuild, a bad argument must not drop the existing index.
    assert main(["index", str(outside), "--rebuild", *project]) == 1
    assert "outside the project" in capsys.readouterr().err
    assert project_store(docs).count() == 4


def test_unreadable_file_is_reported_not_fatal(tmp_path, embedder, store):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "ok.txt").write_text("fine content")
    locked = root / "locked.txt"
    locked.write_text("secret")
    locked.chmod(0)
    try:
        summary = make_indexer(embedder, store, root).index_directory(root)
    finally:
        locked.chmod(0o644)
    assert summary.indexed == 1
    assert [p.name for p, _ in summary.failed] == ["locked.txt"]


def test_rebuild_with_unreachable_ollama_keeps_existing_index(docs, project, monkeypatch):
    main(["index", *project])

    class Down(FakeEmbedder):
        def embed(self, texts):
            raise OllamaError("Cannot connect to Ollama")

    monkeypatch.setattr("mini_rag.cli.OllamaEmbedder", lambda model, host: Down(model))
    assert main(["index", *project, "--rebuild"]) == 1
    assert project_store(docs).count() == 4


def test_invalid_collection_name_is_a_clean_error(project, capsys):
    code = main(["ask", "q", *project, "--collection", "x"])
    assert code == 1 and "Invalid collection name" in capsys.readouterr().err


def test_store_sample_is_reproducible_and_skips_tiny_chunks(tmp_path, embedder):
    store = VectorStore(tmp_path / "db", "docs", embedder.model)
    root = tmp_path / "docs"
    root.mkdir()
    for i in range(6):
        (root / f"f{i}.txt").write_text(f"File {i} says " + f"words about topic number {i}. " * 20)
    (root / "tiny.txt").write_text("short")
    make_indexer(embedder, store, root, size=400, overlap=0).index_directory(root)

    first = store.sample(3, seed=1)
    assert [cid for cid, _ in first] == [cid for cid, _ in store.sample(3, seed=1)]
    assert len(first) == 3
    assert all(c.metadata["source"] != "tiny.txt" for _, c in first)
    assert len(store.sample(100)) == 6
    assert store.chunk_ids() >= {cid for cid, _ in first}


def test_cli_eval_generate_then_score(docs, project, tmp_path, capsys, monkeypatch):
    # Stand-in question writer: first words of the passage, so the fake embedder can find it.
    monkeypatch.setattr(
        "mini_rag.cli.ollama_question_writer",
        lambda model, host, num_ctx: lambda passage: " ".join(passage.split()[:6]) + "?",
    )
    for name in ("long.txt", "other.txt"):
        (docs / name).write_text(f"{name} " + "distinct filler sentence about things. " * 12)
    main(["index", *project])
    capsys.readouterr()

    qfile = tmp_path / "q.jsonl"
    assert main(["eval", str(qfile), "--generate", "2", *project]) == 0
    assert "Wrote 2 questions" in capsys.readouterr().out

    # The set is frozen: refuse to overwrite without --force.
    assert main(["eval", str(qfile), "--generate", "2", *project]) == 1
    assert "already exists" in capsys.readouterr().err

    assert main(["eval", str(qfile), "--k", "5", *project]) == 0
    out = capsys.readouterr().out
    assert "Retrieval eval: 2 questions, k=5" in out and "chunk hit@1" in out


def test_hybrid_finds_exact_identifier_that_dense_ranks_poorly(tmp_path):
    class Blind(FakeEmbedder):
        """Embeddings carry no signal, so only keyword search can find the right chunk."""

        def embed(self, texts):
            return [[1.0] + [0.0] * 63 for _ in texts]

    embedder = Blind()
    store = VectorStore(tmp_path / "db", "docs", embedder.model)
    root = tmp_path / "docs"
    root.mkdir()
    for i in range(30):
        (root / f"n{i}.txt").write_text(f"Generic note {i} about nothing in particular here.")
    (root / "target.txt").write_text("Troubleshooting: error code ZX9041 means the cache is full.")
    make_indexer(embedder, store, root).index_directory(root)

    hybrid = Retriever(embedder, store, top_k=3, mode="hybrid").retrieve("what is ZX9041?")
    assert "target.txt" in [r.chunk.metadata["source"] for r in hybrid]


def test_keyword_index_follows_replacement_and_self_heals(docs, embedder, tmp_path):
    db = tmp_path / "db"
    store = VectorStore(db, "docs", embedder.model)
    indexer = make_indexer(embedder, store, docs)
    indexer.index_directory(docs)
    (docs / "cooking.txt").write_text("Rice should be rinsed before steaming in a covered pot.")
    indexer.index_directory(docs)
    assert [r.chunk.metadata["source"] for r in store.keyword_query("rinsed", 5)] == ["cooking.txt"]
    assert store.keyword_query("pasta", 5) == []  # old version is gone from the keyword index

    store._keywords.clear()  # simulate an interrupted write / a pre-upgrade index
    reopened = VectorStore(db, "docs", embedder.model)
    assert [r.chunk.metadata["source"] for r in reopened.keyword_query("rinsed", 5)] == [
        "cooking.txt"
    ]


def test_prune_removes_only_deleted_files_under_the_directory(docs, embedder, store, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep.txt").write_text("Content from a different indexed directory.")
    indexer = make_indexer(embedder, store, tmp_path)
    indexer.index_directory(docs)
    indexer.index_directory(other)
    before = store.count()

    (docs / "cooking.txt").unlink()
    (other / "keep.txt").unlink()  # deleted too, but outside the directory being pruned
    summary = indexer.index_directory(docs, prune=True)

    assert summary.pruned == 1
    assert store.count() == before - 1
    assert store.keyword_query("pasta", 5) == []
    assert any(
        r.chunk.metadata["source"] == "other/keep.txt" for r in store.keyword_query("different", 5)
    )


def test_prune_without_flag_keeps_deleted_files(docs, embedder, store):
    indexer = make_indexer(embedder, store, docs)
    indexer.index_directory(docs)
    (docs / "cooking.txt").unlink()
    assert indexer.index_directory(docs).pruned == 0
    assert store.count() == 4


def test_prune_keeps_files_that_exist_but_fail_to_load(docs, embedder, store):
    indexer = make_indexer(embedder, store, docs)
    indexer.index_directory(docs)
    (docs / "cooking.txt").write_bytes(b"\xff\xfe\x00bad")  # now unreadable as text
    summary = indexer.index_directory(docs, prune=True)
    assert summary.pruned == 0 and len(summary.failed) == 3
    assert store.count() == 4


def test_prune_refuses_to_wipe_when_directory_has_no_supported_files(docs, embedder, store):
    indexer = make_indexer(embedder, store, docs)
    indexer.index_directory(docs)
    for f in docs.iterdir():
        f.unlink()
    with pytest.raises(MiniRagError, match="Refusing to prune"):
        indexer.index_directory(docs, prune=True)
    assert store.count() == 4


def test_prune_removes_files_that_became_ignored(docs, embedder, store):
    make_indexer(embedder, store, docs).index_directory(docs)
    summary = make_indexer(embedder, store, docs, ignore=["cooking.*"]).index_directory(
        docs, prune=True
    )
    assert summary.pruned == 1 and store.count() == 3


def test_cli_prune_flag(docs, project, capsys):
    main(["index", *project])
    (docs / "cooking.txt").unlink()
    main(["index", "--prune", *project])
    assert "Documents pruned: 1" in capsys.readouterr().out


def test_cli_init_then_use_from_a_subfolder(docs, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("mini_rag.cli.OllamaEmbedder", lambda model, host: FakeEmbedder(model))
    model = ["--embedding-model", "fake-embed"]
    (docs / "sub").mkdir()
    (docs / "sub" / "deep.txt").write_text("Glaciers move slowly down mountain valleys.")
    (docs / "node_modules").mkdir()
    (docs / "node_modules" / "readme.md").write_text("Dependency documentation.")
    assert main(["init", str(docs)]) == 0

    monkeypatch.chdir(docs / "sub")
    main(["index", *model])
    assert "Files discovered: 7" in capsys.readouterr().out  # node_modules is ignored
    assert main(["status", *model]) == 0
    out = capsys.readouterr().out
    assert f"Project: {docs.resolve()}" in out and "Indexed chunks: 5" in out
    assert "deep.txt" in {Path(s).name for s in project_store(docs).indexed_documents().values()}
    assert "sub/deep.txt" in project_store(docs).indexed_documents().values()


def test_cli_indexing_a_subfolder_keeps_project_relative_sources(docs, project, capsys):
    (docs / "sub").mkdir()
    (docs / "sub" / "deep.txt").write_text("Glaciers move slowly down mountain valleys.")
    assert main(["index", str(docs / "sub"), *project]) == 0
    assert list(project_store(docs).indexed_documents().values()) == ["sub/deep.txt"]


def test_moved_project_keeps_its_index(docs, project, tmp_path, capsys):
    main(["index", *project])
    moved = tmp_path / "renamed"
    docs.rename(moved)
    capsys.readouterr()
    main(["index", "--project", str(moved), "--embedding-model", "fake-embed"])
    out = capsys.readouterr().out
    assert "Files unchanged: 4" in out and "Files indexed: 0" in out
    assert project_store(moved).count() == 4


def test_cli_without_a_project_explains_what_to_do(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["ask", "q"]) == 1
    assert "mini-rag init" in capsys.readouterr().err
    assert not (tmp_path / ".mini-rag").exists()  # never created silently


def test_dotenv_files_are_not_loaded(docs, project, capsys, monkeypatch):
    (docs / ".env").write_text("RAG_TOP_K=not-a-number\n")
    monkeypatch.chdir(docs)
    assert main(["status", *project]) == 0


def test_cwd_dotenv_cannot_reconfigure_chroma(tmp_path, monkeypatch):
    # The Python backend would unpickle files found in the database directory.
    (tmp_path / ".env").write_text(
        "chroma_api_impl=chromadb.api.segment.SegmentAPI\nchroma_server_host=attacker.example\n"
    )
    monkeypatch.chdir(tmp_path)
    store = VectorStore(tmp_path / "db", "documents", "fake-embed")
    settings = store._client.get_settings()
    assert settings.chroma_api_impl == "chromadb.api.rust.RustBindingsAPI"
    assert settings.chroma_server_host is None


def test_rebuild_of_a_subfolder_is_refused(docs, project, capsys):
    (docs / "sub").mkdir()
    (docs / "sub" / "deep.txt").write_text("Glaciers move slowly down mountain valleys.")
    main(["index", *project])
    capsys.readouterr()

    assert main(["index", str(docs / "sub"), "--rebuild", *project]) == 1

    assert "whole project" in capsys.readouterr().err
    assert project_store(docs).count() == 5


def test_rebuild_naming_the_project_folder_is_allowed(docs, project, capsys):
    main(["index", *project])
    capsys.readouterr()
    main(["index", str(docs), "--rebuild", *project])
    assert "Files indexed: 4" in capsys.readouterr().out


@pytest.mark.parametrize("prune", [False, True])
def test_emptied_files_stop_answering_with_old_content(docs, embedder, store, prune):
    indexer = make_indexer(embedder, store, docs)
    indexer.index_directory(docs)
    (docs / "cooking.txt").write_text("   \n")
    make_pdf(docs / "birds.pdf", [""])  # e.g. replaced by a scan without a text layer

    summary = indexer.index_directory(docs, prune=prune)

    assert store.count() == 2
    assert store.keyword_query("pasta penguins", 5) == []
    reasons = {path.name: reason for path, reason in summary.failed}
    assert "removed" in reasons["cooking.txt"] and "removed" in reasons["birds.pdf"]
    assert "removed" not in reasons["empty.txt"]  # it never had any content


def test_unparseable_files_keep_their_old_content(docs, embedder, store):
    indexer = make_indexer(embedder, store, docs)
    indexer.index_directory(docs)
    (docs / "birds.pdf").write_bytes(b"half written garbage")
    indexer.index_directory(docs, prune=True)
    assert store.count() == 4
    assert store.keyword_query("penguins", 5)


@pytest.mark.parametrize(
    "config,flags,expected",
    [
        ("", [], False),
        ("RAG_THINK=true", [], True),
        ("RAG_THINK=true", ["--no-think"], False),
        ("", ["--think"], True),
    ],
)
def test_thinking_reaches_the_generator(docs, project, monkeypatch, config, flags, expected):
    monkeypatch.delenv("RAG_THINK", raising=False)
    (docs / ".mini-rag" / "config.env").write_text(config)
    seen = {}

    def fake_generator(model, host, system_prompt, num_ctx, think):
        seen["think"] = think
        return MagicMock()

    monkeypatch.setattr("mini_rag.cli.Generator", fake_generator)
    assert main(["ask", "q", *project, *flags]) == 0
    assert seen["think"] is expected


def test_small_chunk_size_works_without_naming_an_overlap(docs, project, capsys):
    main(["index", "--chunk-size", "50", *project])
    assert "Files indexed: 4" in capsys.readouterr().out


@pytest.mark.parametrize("flag", [["--chunk-size", "50"], ["--chunk-overlap", "10"]])
def test_one_off_chunk_flags_say_how_to_make_them_permanent(docs, project, capsys, flag):
    main(["index", *flag, *project])
    err = capsys.readouterr().err
    assert "this run only" in err and "config.env" in err


def test_chunk_settings_from_config_need_no_note(docs, project, capsys):
    (docs / ".mini-rag" / "config.env").write_text("RAG_CHUNK_SIZE=50\n")
    main(["index", *project])
    assert "this run only" not in capsys.readouterr().err
    main(["index", *project])
    assert "Files unchanged: 4" in capsys.readouterr().out
