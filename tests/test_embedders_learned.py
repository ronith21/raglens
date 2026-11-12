"""Tests for the LSA and word-vector embedders, the factory, and fitted-index persistence."""

import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from raglens.config import Settings
from raglens.embed import factory
from raglens.embed.factory import build_embedder, embedder_label
from raglens.embed.hash import HashEmbedder
from raglens.embed.lsa import LSAEmbedder
from raglens.embed.word_vectors import WordVectorEmbedder
from raglens.ingest.models import Chunk
from raglens.pipeline.builder import build_retriever
from raglens.retrieve.store import IndexStore

TEXTS = [
    "gradient descent updates parameters using the gradient of the loss and a learning rate",
    "a large learning rate makes gradient descent overshoot and diverge",
    "bm25 ranks documents by term frequency and inverse document frequency",
    "inverse document frequency rewards rare terms in bm25 lexical search",
    "dense embeddings place similar sentences close together in vector space",
    "cosine similarity between embeddings measures semantic closeness of sentences",
]


def test_lsa_requires_fit_and_normalises() -> None:
    """Unfitted use fails loudly; fitted vectors are unit length."""
    emb = LSAEmbedder(n_components=4)
    assert not emb.is_fitted
    with pytest.raises(RuntimeError, match="fitted"):
        emb.embed_texts(["x"])
    emb.fit(TEXTS)
    mat = emb.embed_texts(TEXTS)
    assert mat.shape == (6, emb.dimension)
    np.testing.assert_allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)
    assert emb.embed_texts([]).shape == (0, emb.dimension)


def test_lsa_groups_cooccurring_topics() -> None:
    """A query about one topic is closer to that topic's sentences than to other topics."""
    emb = LSAEmbedder(n_components=3).fit(TEXTS)
    q = emb.embed_query("learning rate for gradient descent")
    sims = emb.embed_texts(TEXTS) @ q
    assert int(np.argmax(sims)) in (0, 1)
    assert sims[:2].mean() > sims[2:].mean()


def test_lsa_unknown_words_map_to_zero_and_validation() -> None:
    """Out-of-vocabulary text gives a zero vector; bad arguments raise."""
    emb = LSAEmbedder(n_components=3).fit(TEXTS)
    assert float(np.linalg.norm(emb.embed_query("zzzz qqqq"))) == 0.0
    with pytest.raises(ValueError):
        LSAEmbedder(n_components=1)
    with pytest.raises(ValueError):
        LSAEmbedder().fit(["only one"])
    with pytest.raises(RuntimeError):
        LSAEmbedder().save_state("/tmp/never")


def test_lsa_state_round_trip(tmp_path: Path) -> None:
    """Saved and restored LSA produces identical vectors."""
    emb = LSAEmbedder(n_components=4).fit(TEXTS)
    emb.save_state(tmp_path)
    restored = LSAEmbedder()
    assert restored.has_saved_state(tmp_path)
    restored.load_state(tmp_path)
    np.testing.assert_allclose(restored.embed_texts(TEXTS), emb.embed_texts(TEXTS), atol=1e-6)


def test_index_store_restores_fitted_embedder_without_reembedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading an LSA index must not refit or re-embed the corpus."""
    chunks = [
        Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=t, metadata={"source": f"s{i}"})
        for i, t in enumerate(TEXTS)
    ]
    retriever = build_retriever(chunks, LSAEmbedder(n_components=4), Settings())
    before = [r.chunk.chunk_id for r in retriever.retrieve("bm25 inverse document frequency", 3)]
    IndexStore.save(retriever, tmp_path)

    def fail_fit(self: LSAEmbedder, texts: list[str]) -> LSAEmbedder:
        raise AssertionError("fit must not be called when loading an index")

    monkeypatch.setattr(LSAEmbedder, "fit", fail_fit)
    loaded = IndexStore.load(tmp_path, embedder=LSAEmbedder())
    after = [r.chunk.chunk_id for r in loaded.retrieve("bm25 inverse document frequency", 3)]
    assert before == after


def test_attach_index_validates_shapes(tmp_path: Path) -> None:
    """A FAISS index that does not match the chunks or the embedder is rejected."""
    chunks = [Chunk(chunk_id=f"c{i}", doc_id="d", text=t) for i, t in enumerate(TEXTS)]
    retriever = build_retriever(chunks, HashEmbedder(dimension=16), Settings())
    index = retriever.dense_retriever.index
    assert index is not None
    with pytest.raises(ValueError, match="holds"):
        retriever.dense_retriever.attach_index(chunks[:2], index)
    other = build_retriever(chunks, HashEmbedder(dimension=8), Settings())
    with pytest.raises(ValueError, match="dimension"):
        other.dense_retriever.attach_index(chunks, index)


class _FakeLexeme:
    def __init__(self, vector: np.ndarray | None) -> None:
        self.has_vector = vector is not None
        self.vector = vector if vector is not None else np.zeros(3)


class _FakeVocab:
    """Three-dimensional toy word vectors: animals, finance, and a shared 'common' axis."""

    table: ClassVar[dict[str, list[float]]] = {
        "dog": [1.0, 0.0, 1.0],
        "puppy": [0.9, 0.1, 1.0],
        "cat": [0.8, 0.0, 1.0],
        "bank": [0.0, 1.0, 1.0],
        "loan": [0.1, 0.9, 1.0],
        "money": [0.0, 0.8, 1.0],
    }

    def __getitem__(self, word: str) -> _FakeLexeme:
        vec = self.table.get(word)
        return _FakeLexeme(None if vec is None else np.asarray(vec, dtype=np.float32))


def _toy_word_vectors() -> WordVectorEmbedder:
    emb = WordVectorEmbedder()
    emb._vocab = _FakeVocab()
    emb._dim = 3
    return emb


def test_word_vectors_capture_synonyms_without_shared_words() -> None:
    """'puppy' finds the dog sentence although they share no word: the point of embeddings."""
    emb = _toy_word_vectors().fit(["dog cat", "bank loan", "money bank"])
    docs = emb.embed_texts(["dog cat", "bank loan"])
    q = emb.embed_query("puppy")
    assert float(docs[0] @ q) > float(docs[1] @ q)
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    assert float(np.linalg.norm(emb.embed_query("unknownword"))) == 0.0
    assert emb.dimension == 3 and emb.is_fitted


def test_word_vectors_state_round_trip(tmp_path: Path) -> None:
    """IDF weights and the removed common direction persist."""
    emb = _toy_word_vectors().fit(["dog cat", "bank loan", "money bank"])
    emb.save_state(tmp_path)
    restored = _toy_word_vectors()
    assert restored.has_saved_state(tmp_path)
    restored.load_state(tmp_path)
    np.testing.assert_allclose(restored.embed_texts(["dog loan"]), emb.embed_texts(["dog loan"]))
    with pytest.raises(RuntimeError):
        _toy_word_vectors().save_state(tmp_path / "x")


def test_word_vectors_load_real_model_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spaCy model is loaded on first use and an empty vector table is rejected."""
    fake_spacy = types.ModuleType("spacy")

    class _Vectors:
        shape = (0, 300)

    class _NLP:
        vocab = types.SimpleNamespace(vectors=_Vectors())

    fake_spacy.load = lambda *a, **k: _NLP()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    with pytest.raises(RuntimeError, match="no word vectors"):
        _ = WordVectorEmbedder().dimension


def test_factory_choices_and_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory builds each embedder and falls back minilm -> glove -> lsa, never to hash."""
    assert isinstance(build_embedder("hash", hash_dim=8), HashEmbedder)
    assert embedder_label(build_embedder("hash")) == "hash"
    assert embedder_label(build_embedder("lsa")) == "lsa"
    with pytest.raises(ValueError, match="Unknown embedder"):
        build_embedder("word2vec")

    def broken_st(*args: object, **kwargs: object) -> None:
        raise RuntimeError("offline")

    st_module = types.ModuleType("sentence_transformers")
    st_module.SentenceTransformer = broken_st  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_module)
    monkeypatch.setattr(
        WordVectorEmbedder, "_load", lambda self: (_ for _ in ()).throw(OSError("no model"))
    )
    assert isinstance(build_embedder("minilm"), LSAEmbedder)

    monkeypatch.setattr(WordVectorEmbedder, "_load", lambda self: None)
    monkeypatch.setattr(WordVectorEmbedder, "dimension", property(lambda self: 300))
    fallback = factory.build_embedder("minilm")
    assert isinstance(fallback, WordVectorEmbedder)
    assert embedder_label(fallback) == "glove"
