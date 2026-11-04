"""Unit tests for embedding models: HashEmbedder and SentenceTransformerEmbedder."""

import sys
import types

import numpy as np
import pytest

from raglens.embed.hash import HashEmbedder
from raglens.embed.sentence_transformer import SentenceTransformerEmbedder


def _fake_sentence_transformers(factory: object) -> types.ModuleType:
    """Build a stand-in ``sentence_transformers`` module whose model class is ``factory``."""
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = factory  # type: ignore[attr-defined]
    return module


def test_hash_embedder_dimension_and_shape() -> None:
    """HashEmbedder initializes with proper dimension and outputs expected shapes."""
    embedder = HashEmbedder(dimension=128)
    assert embedder.dimension == 128
    assert embedder.name == "hash-128"

    mat = embedder.embed_texts(["hello world", "retrieval augmented generation", "ml"])
    assert mat.shape == (3, 128)
    assert mat.dtype == np.float32


def test_hash_embedder_invalid_dimension() -> None:
    """Non-positive dimension raises ValueError."""
    with pytest.raises(ValueError, match="dimension must be positive"):
        HashEmbedder(dimension=0)


def test_hash_embedder_determinism() -> None:
    """HashEmbedder must be 100% deterministic across different instances and calls."""
    embedder1 = HashEmbedder(dimension=64)
    embedder2 = HashEmbedder(dimension=64)

    text = "Gradient descent with momentum and weight decay."
    v1 = embedder1.embed_query(text)
    np.testing.assert_array_almost_equal(v1, embedder2.embed_query(text))
    np.testing.assert_array_almost_equal(v1, embedder1.embed_query(text))


def test_hash_embedder_normalization() -> None:
    """Vectors produced by HashEmbedder should be unit normalized."""
    embedder = HashEmbedder(dimension=64)
    texts = ["test query", "", "another complex sentence with numbers 12345", "the of and"]
    mat = embedder.embed_texts(texts)
    for row in mat:
        assert np.isclose(float(np.linalg.norm(row)), 1.0, atol=1e-5)


def test_hash_embedder_ignores_stopwords_and_inflection() -> None:
    """Stopwords do not affect the vector and plural/derived forms map together."""
    embedder = HashEmbedder(dimension=64)
    a = embedder.embed_query("What is the learning rate?")
    b = embedder.embed_query("learning rates")
    np.testing.assert_array_almost_equal(a, b)
    unrelated = embedder.embed_query("isolation forest anomalies")
    assert float(a @ b) > float(a @ unrelated)


def test_hash_embedder_empty_list() -> None:
    """Empty list returns empty 2D array."""
    assert HashEmbedder(dimension=32).embed_texts([]).shape == (0, 32)


def test_sentence_transformer_embedder_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """SentenceTransformerEmbedder smoothly falls back to HashEmbedder on failure."""

    def failing_model(*args: object, **kwargs: object) -> None:
        raise ImportError("Simulated missing sentence-transformers")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", _fake_sentence_transformers(failing_model)
    )
    embedder = SentenceTransformerEmbedder(model_name="nonexistent-model", fallback_to_hash=True)
    embedder._load_model()
    assert embedder._fallback_embedder is not None
    assert embedder.is_fallback
    assert embedder.dimension == 384
    assert "fallback" in embedder.name
    assert embedder.embed_texts(["hello fallback"]).shape == (1, 384)


def test_sentence_transformer_embedder_raises_when_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When fallback is disabled, loading failure raises exception."""

    def failing_model(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Model download failed")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", _fake_sentence_transformers(failing_model)
    )
    embedder = SentenceTransformerEmbedder(model_name="nonexistent-model", fallback_to_hash=False)
    with pytest.raises(RuntimeError, match="Model download failed"):
        embedder._load_model()


def test_sentence_transformer_embedder_uses_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the package loads, encode() output is used and normalized flags are passed."""

    class StubModel:
        def __init__(self, name: str, device: str) -> None:
            self.calls: list[dict[str, object]] = []

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            self.calls.append(kwargs)
            return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", _fake_sentence_transformers(StubModel)
    )
    embedder = SentenceTransformerEmbedder(model_name="stub")
    assert embedder.dimension == 8
    assert not embedder.is_fallback
    assert embedder.name == "stub"
    out = embedder.embed_texts(["a", "b"])
    assert out.shape == (2, 8)
    assert embedder._model.calls[-1]["normalize_embeddings"] is True
    assert embedder.embed_texts([]).shape == (0, 8)
