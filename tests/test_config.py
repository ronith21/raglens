"""Tests for configuration settings."""

import pytest

from raglens.config import Settings


def test_settings_default() -> None:
    """Settings load with sensible defaults."""
    s = Settings(_env_file=None)
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.embedding_device == "cpu"
    assert s.default_top_k == 5
    assert s.rrf_k == 60
    assert s.llm_provider in ("fake", "ollama", "openai")
    assert 0.0 <= s.min_query_coverage <= 1.0


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables override defaults (case-insensitive names)."""
    monkeypatch.setenv("RRF_K", "30")
    monkeypatch.setenv("USE_HASH_EMBEDDER", "true")
    s = Settings(_env_file=None)
    assert s.rrf_k == 30
    assert s.use_hash_embedder is True
