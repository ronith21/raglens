"""Latent Semantic Analysis (LSA) embedder: a dense embedding learned from the corpus itself.

LSA builds a TF-IDF matrix over the indexed chunks and factorises it with a truncated SVD.
Each chunk (and each query) is projected onto the top singular vectors, giving a dense,
L2-normalised vector. Words that co-occur across chunks ("embedding", "vector", "cosine")
end up pointing in similar directions, so a query can match a chunk that does not share its
exact words. It is fully offline and deterministic, which makes it the default fallback
when a neural model such as ``all-MiniLM-L6-v2`` cannot be downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from raglens.embed.base import FittableEmbedder

_STATE_FILE = "lsa_embedder.npz"
_META_FILE = "lsa_embedder.json"


class LSAEmbedder(FittableEmbedder):
    """TF-IDF + truncated SVD embedder that must be fitted on the corpus before use."""

    def __init__(self, n_components: int = 128, random_state: int = 42) -> None:
        if n_components <= 1:
            raise ValueError(f"n_components must be greater than 1, got {n_components}")
        self.n_components = n_components
        self.random_state = random_state
        self._vectorizer: TfidfVectorizer | None = None
        self._components: npt.NDArray[np.float32] | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` (or :meth:`load`) has been called."""
        return self._vectorizer is not None and self._components is not None

    @property
    def dimension(self) -> int:
        """Number of latent dimensions actually learned (may be below ``n_components``)."""
        if self._components is None:
            return self.n_components
        return int(self._components.shape[0])

    @staticmethod
    def _new_vectorizer() -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b",
        )

    def fit(self, texts: list[str]) -> LSAEmbedder:
        """Learn the vocabulary, IDF weights and latent topic directions from ``texts``."""
        if len(texts) < 2:
            raise ValueError("LSAEmbedder needs at least 2 texts to fit")
        vectorizer = self._new_vectorizer()
        tfidf = vectorizer.fit_transform(texts)
        max_rank = min(tfidf.shape[0], tfidf.shape[1]) - 1
        k = max(2, min(self.n_components, max_rank))
        svd = TruncatedSVD(n_components=k, random_state=self.random_state, algorithm="arpack")
        svd.fit(tfidf)
        self._vectorizer = vectorizer
        self._components = np.asarray(svd.components_, dtype=np.float32)
        return self

    def embed_texts(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Project texts into the latent space and L2-normalise them."""
        if self._vectorizer is None or self._components is None:
            raise RuntimeError("LSAEmbedder must be fitted before embedding texts")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        tfidf = self._vectorizer.transform(texts)
        dense = np.asarray(tfidf @ self._components.T, dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        # A text with no known vocabulary maps to the zero vector (cosine 0 to everything).
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(dense / norms, dtype=np.float32)

    def save_state(self, directory: str | Path) -> None:
        """Persist the fitted state as plain arrays and JSON (no pickle)."""
        if self._vectorizer is None or self._components is None:
            raise RuntimeError("Cannot save an unfitted LSAEmbedder")
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        vocab = self._vectorizer.vocabulary_
        terms = sorted(vocab, key=lambda t: vocab[t])
        np.savez_compressed(
            dir_path / _STATE_FILE,
            idf=np.asarray(self._vectorizer.idf_, dtype=np.float64),
            components=self._components,
        )
        meta: dict[str, Any] = {
            "n_components": self.n_components,
            "random_state": self.random_state,
            "terms": terms,
        }
        (dir_path / _META_FILE).write_text(json.dumps(meta), encoding="utf-8")

    def load_state(self, directory: str | Path) -> None:
        """Restore state written by :meth:`save_state`."""
        dir_path = Path(directory)
        meta = json.loads((dir_path / _META_FILE).read_text(encoding="utf-8"))
        with np.load(dir_path / _STATE_FILE) as arrays:
            idf = arrays["idf"]
            components = arrays["components"]
        self.n_components = int(meta["n_components"])
        self.random_state = int(meta["random_state"])
        vectorizer = self._new_vectorizer()
        vectorizer.vocabulary_ = {term: i for i, term in enumerate(meta["terms"])}
        vectorizer.idf_ = idf
        self._vectorizer = vectorizer
        self._components = np.asarray(components, dtype=np.float32)

    def has_saved_state(self, directory: str | Path) -> bool:
        """Return True if a saved LSA state exists in ``directory``."""
        return (Path(directory) / _STATE_FILE).exists()
