"""Shared fixtures: an offline engine over the real sample corpus (HashEmbedder + FakeLLM)."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from raglens.config import Settings
from raglens.embed.hash import HashEmbedder
from raglens.engine import RAGEngine
from raglens.llm.fake import FakeLLM

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "sample_corpus"
EVAL_DATASET = REPO_ROOT / "data" / "eval" / "qa.jsonl"


@pytest.fixture(autouse=True)
def _no_network_models(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Guarantee tests never try to download models, even if the package is installed."""
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


def make_settings(tmp_path: Path, corpus_dir: Path = CORPUS_DIR, **overrides: object) -> Settings:
    """Settings isolated from any local .env and writing the index under ``tmp_path``."""
    values: dict[str, object] = {
        "use_hash_embedder": True,
        "llm_provider": "fake",
        "corpus_dir": str(corpus_dir),
        "eval_dataset": str(EVAL_DATASET),
        "index_dir": str(tmp_path / "index"),
        "query_log_path": str(tmp_path / "logs" / "queries.db"),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline settings over the sample corpus."""
    return make_settings(tmp_path)


@pytest.fixture
def engine(settings: Settings) -> RAGEngine:
    """An engine with the sample corpus indexed."""
    built = RAGEngine(settings, embedder=HashEmbedder(dimension=384), llm=FakeLLM())
    built.load_or_build()
    return built
