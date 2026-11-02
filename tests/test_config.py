"""Tests for configuration settings."""

from raglens.config import Settings


def test_settings_default() -> None:
    """Settings load with sensible defaults."""
    s = Settings()
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.embedding_device == "cpu"
    assert s.default_top_k == 5
    assert s.rrf_k == 60
    assert s.llm_provider in ("fake", "ollama", "openai")
