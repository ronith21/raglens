"""Unit tests for embedding models: HashEmbedder and SentenceTransformerEmbedder."""

import numpy as np
import pytest

from raglens.embed.hash import HashEmbedder
from raglens.embed.sentence_transformer import SentenceTransformerEmbedder


def test_hash_embedder_dimension_and_shape() -> None:
    """HashEmbedder initializes with proper dimension and outputs expected shapes."""
    embedder = HashEmbedder(dimension=128)
    assert embedder.dimension == 128

    texts = ["hello world", "retrieval augmented generation", "machine learning"]
    mat = embedder.embed_texts(texts)
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
    v2 = embedder2.embed_query(text)
    v3 = embedder1.embed_query(text)

    np.testing.assert_array_almost_equal(v1, v2)
    np.testing.assert_array_almost_equal(v1, v3)


def test_hash_embedder_normalization() -> None:
    """Vectors produced by HashEmbedder should be unit normalized."""
    embedder = HashEmbedder(dimension=64)
    texts = ["test query", "", "another complex sentence with numbers 12345"]
    mat = embedder.embed_texts(texts)

    for i in range(len(texts)):
        norm = float(np.linalg.norm(mat[i]))
        assert np.isclose(norm, 1.0, atol=1e-5)


def test_hash_embedder_empty_list() -> None:
    """Empty list returns empty 2D array."""
    embedder = HashEmbedder(dimension=32)
    mat = embedder.embed_texts([])
    assert mat.shape == (0, 32)


def test_sentence_transformer_embedder_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """SentenceTransformerEmbedder smoothly falls back to HashEmbedder on failure."""

    def mock_import(*args: object, **kwargs: object) -> None:
        raise ImportError("Simulated missing sentence-transformers")

    embedder = SentenceTransformerEmbedder(model_name="nonexistent-model", fallback_to_hash=True)
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        mock_import,
        raising=False,
    )
    embedder._load_model()
    assert embedder._fallback_embedder is not None
    assert embedder.dimension == 384

    res = embedder.embed_texts(["hello fallback"])
    assert res.shape == (1, 384)


def test_sentence_transformer_embedder_raises_when_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When fallback is disabled, loading failure raises exception."""
    embedder = SentenceTransformerEmbedder(model_name="nonexistent-model", fallback_to_hash=False)

    def mock_fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Model download failed")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", mock_fail, raising=False)
    with pytest.raises(RuntimeError, match="Model download failed"):
        embedder._load_model()
