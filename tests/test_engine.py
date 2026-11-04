"""Tests for the application engine (index lifecycle and configuration)."""

from pathlib import Path

import pytest

from raglens.embed.hash import HashEmbedder
from raglens.embed.sentence_transformer import SentenceTransformerEmbedder
from raglens.engine import RAGEngine, create_embedder
from raglens.llm.fake import FakeLLM
from raglens.llm.ollama import OllamaLLM
from tests.conftest import make_settings


def test_builds_index_from_corpus_and_persists(engine: RAGEngine) -> None:
    """The sample corpus is ingested, chunked and saved to INDEX_DIR."""
    docs = engine.documents()
    assert len(docs) == 20
    assert docs[0].title.startswith("Gradient Descent")
    assert sum(d.num_chunks for d in docs) == len(engine.chunks)
    assert (Path(engine.settings.index_dir) / "metadata.json").exists()


def test_reload_uses_persisted_index(engine: RAGEngine) -> None:
    """A fresh engine on the same settings loads instead of re-ingesting."""
    again = RAGEngine(engine.settings, embedder=HashEmbedder(dimension=384), llm=FakeLLM())
    again.load_or_build()
    assert [c.chunk_id for c in again.chunks] == [c.chunk_id for c in engine.chunks]


def test_ingest_is_idempotent_and_adds_new_files(engine: RAGEngine, tmp_path: Path) -> None:
    """Re-ingesting known files adds nothing; a new file is indexed and searchable."""
    before = len(engine.chunks)
    report = engine.ingest_paths([Path(engine.settings.corpus_dir)])
    assert report.documents_added == 0 and report.documents_skipped == 20
    assert len(engine.chunks) == before

    note = tmp_path / "zebra.md"
    note.write_text("# Zebra Facts\n\nThe okapi zebracorn eats quasar berries.", encoding="utf-8")
    report = engine.ingest_paths([note])
    assert report.documents_added == 1 and report.total_documents == 21
    answer = engine.ask("What does the zebracorn eat?", retriever="bm25")
    assert "quasar" in answer.answer


def test_ingest_rejects_unsupported_file(engine: RAGEngine, tmp_path: Path) -> None:
    """Unsupported files are refused with a clear error."""
    bad = tmp_path / "data.csv"
    bad.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported path"):
        engine.ingest_paths([bad])


def test_rebuild_discards_existing_index(engine: RAGEngine, tmp_path: Path) -> None:
    """rebuild=True replaces the index with only the given paths."""
    note = tmp_path / "only.txt"
    note.write_text("Only document in the rebuilt index.", encoding="utf-8")
    report = engine.ingest_paths([note], rebuild=True)
    assert report.total_documents == 1


def test_missing_corpus_leaves_empty_index(tmp_path: Path) -> None:
    """A missing corpus directory is tolerated (empty index, warning logged)."""
    settings = make_settings(tmp_path, corpus_dir=tmp_path / "missing")
    empty = RAGEngine(settings, embedder=HashEmbedder(dimension=16))
    empty.load_or_build()
    assert empty.chunks == []


@pytest.mark.parametrize("retriever", ["bm25", "dense", "hybrid"])
def test_every_retriever_answers_with_citations(engine: RAGEngine, retriever: str) -> None:
    """All three retrievers produce a cited answer for a corpus question."""
    answer = engine.ask("How does SMOTE create synthetic minority samples?", retriever=retriever)
    assert not answer.refused
    assert answer.citations
    assert answer.retriever == retriever


def test_rerank_option_and_unknown_retriever(engine: RAGEngine) -> None:
    """rerank=True adds the reranker (pass-through offline); bad names raise."""
    answer = engine.ask("What is an inverted index?", rerank=True)
    assert answer.reranked and answer.retriever == "hybrid+rerank"
    assert "rerank" in answer.timings_ms
    with pytest.raises(ValueError, match="Unknown retriever"):
        engine.retriever("tfidf")


def test_llm_override_and_embedder_factory(engine: RAGEngine, tmp_path: Path) -> None:
    """Provider overrides return a fresh client; the embedder follows settings."""
    assert isinstance(engine.llm("ollama"), OllamaLLM)
    assert engine.llm() is engine.llm("fake")
    assert isinstance(create_embedder(make_settings(tmp_path)), HashEmbedder)
    real = create_embedder(make_settings(tmp_path, use_hash_embedder=False))
    assert isinstance(real, SentenceTransformerEmbedder)
    assert real.is_fallback  # sentence-transformers is blocked in tests


def test_multi_query_pipeline_and_rewriter_refresh(engine: RAGEngine, tmp_path: Path) -> None:
    """multi_query wraps the retriever; the acronym glossary is rebuilt after ingest."""
    answer = engine.ask("What is RRF?", multi_query=True)
    assert answer.retriever == "hybrid+mq"
    assert engine.rewriter.acronyms["rrf"] == "reciprocal rank fusion"
    assert engine.rewriter is engine.rewriter  # cached while the index is unchanged

    note = tmp_path / "zz.md"
    note.write_text("# Zz\n\nQuantum Widget Engine (QWE) powers zebracorns.", encoding="utf-8")
    engine.ingest_paths([note])
    assert engine.rewriter.acronyms["qwe"] == "quantum widget engine"
