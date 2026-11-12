"""CLI tests via Typer's CliRunner with offline settings from environment variables."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from raglens.cli.main import app

from .conftest import CORPUS, QA

runner = CliRunner()


@pytest.fixture()
def offline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every setting at the sample data and a temporary index."""
    monkeypatch.setenv("CORPUS_DIR", str(CORPUS))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("EMBEDDER", "lsa")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EVAL_DATASET", str(QA))
    return tmp_path


def test_ingest_then_ask(offline_env: Path) -> None:
    """Ingest writes an index; ask loads it and prints a cited answer."""
    r = runner.invoke(app, ["ingest"])
    assert r.exit_code == 0, r.output
    assert "Indexed 20 documents" in r.output
    assert (offline_env / "index" / "metadata.json").exists()
    assert (offline_env / "index" / "lsa_embedder.npz").exists()

    r = runner.invoke(app, ["ask", "How does Byte-Pair Encoding build its vocabulary?"])
    assert r.exit_code == 0, r.output
    assert "[1]" in r.output and "07_tokenization_algorithms" in r.output

    r = runner.invoke(app, ["ask", "--json", "Who won the 2018 FIFA World Cup?"])
    assert '"refused": true' in r.output


def test_ask_rejects_unknown_retriever(offline_env: Path) -> None:
    """Typos in --retriever fail fast."""
    r = runner.invoke(app, ["ask", "q", "--retriever", "sparse"])
    assert r.exit_code != 0


def test_eval_writes_report(offline_env: Path) -> None:
    """A reduced evaluation writes the Markdown report."""
    out = offline_env / "proof"
    args = [
        "eval",
        "--out-dir",
        str(out),
        "--embedders",
        "lsa",
        "--main-embedder",
        "lsa",
        "--chunk-sizes",
        "512",
    ]
    r = runner.invoke(app, [*args, "--main-chunk-size", "512"])
    assert r.exit_code == 0, r.output
    assert "hybrid: MRR=" in r.output
    assert (out / "eval_report.md").exists()


def test_eval_validates_main_choices(offline_env: Path) -> None:
    """The headline configuration must be part of the ablation."""
    r = runner.invoke(app, ["eval", "--embedders", "lsa", "--main-embedder", "glove"])
    assert r.exit_code != 0
    r = runner.invoke(
        app, ["eval", "--embedders", "lsa", "--main-embedder", "lsa", "--chunk-sizes", "256"]
    )
    assert r.exit_code != 0
