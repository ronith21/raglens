"""Tests for the Typer CLI."""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from raglens.cli.main import app
from tests.conftest import CORPUS_DIR, EVAL_DATASET

runner = CliRunner()


@pytest.fixture(autouse=True)
def offline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every command at the sample corpus with a temp index and offline components."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_HASH_EMBEDDER", "true")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("CORPUS_DIR", str(CORPUS_DIR))
    monkeypatch.setenv("EVAL_DATASET", str(EVAL_DATASET))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))


def test_ingest_then_ask() -> None:
    """ingest builds the index; ask prints a cited answer and sources."""
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0, result.output
    assert "Added 20 documents" in result.output

    result = runner.invoke(app, ["ask", "What is an inverted index?"])
    assert result.exit_code == 0, result.output
    assert "postings list" in result.output
    assert re.search(r"\[\d\] .*11_lexical_search_bm25\.md", result.output)

    result = runner.invoke(
        app, ["ask", "What does the b parameter do in BM25?", "--retriever", "bm25"]
    )
    assert "document length" in result.output


def test_ask_json_and_refusal() -> None:
    """--json emits a parseable response; off-topic questions are refused."""
    result = runner.invoke(app, ["ask", "What is the capital of Australia?", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["refused"] is True


def test_ask_rejects_unknown_retriever() -> None:
    """Invalid --retriever values fail with a usage error."""
    result = runner.invoke(app, ["ask", "q", "--retriever", "tfidf"])
    assert result.exit_code != 0


def test_ask_reports_llm_errors() -> None:
    """A provider without credentials exits non-zero with the reason."""
    result = runner.invoke(app, ["ask", "How does BPE work?", "--provider", "openai"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_ingest_missing_path() -> None:
    """Missing paths are reported and exit 1."""
    result = runner.invoke(app, ["ingest", "does-not-exist"])
    assert result.exit_code == 1
    assert "Not found" in result.output


def test_eval_single_config() -> None:
    """eval prints a JSON summary for one configuration."""
    result = runner.invoke(app, ["eval", "--retriever", "bm25"])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["config"] == "bm25"
    assert summary["retrieval"]["hit@5"] > 0.9


def test_synth_writes_jsonl(tmp_path: Path) -> None:
    """synth writes the requested number of synthetic questions."""
    out = tmp_path / "syn.jsonl"
    result = runner.invoke(app, ["synth", "--out", str(out), "--num", "5"])
    assert result.exit_code == 0, result.output
    assert len(out.read_text(encoding="utf-8").splitlines()) == 5
