"""Deterministic, dependency-free hash embedder for testing and offline fallbacks."""

import hashlib
import re

import numpy as np
import numpy.typing as npt

from raglens.embed.base import BaseEmbedder
from raglens.textutil import content_stems

_TOKEN_RE = re.compile(r"\w+")


class HashEmbedder(BaseEmbedder):
    """Deterministic hash-based embedder producing normalized pseudo-random vectors.

    Each stemmed, lowercase content word is mapped to a fixed random Gaussian vector seeded by
    its SHA-256 hash, and a text is the normalized sum of its token vectors. Texts sharing many
    tokens therefore get a high cosine similarity: a cheap bag-of-words embedding with no model
    download, which is what tests and offline environments need.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dim = dimension
        self._cache: dict[str, npt.NDArray[np.float32]] = {}

    @property
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""
        return self._dim

    @property
    def name(self) -> str:
        """Identifier including the dimension."""
        return f"hash-{self._dim}"

    @staticmethod
    def _hash_token(token: str) -> int:
        """Hash a single token string to an integer seed using SHA256."""
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _token_vector(self, token: str) -> npt.NDArray[np.float32]:
        """Return the (cached) random vector for one token."""
        vec = self._cache.get(token)
        if vec is None:
            rng = np.random.default_rng(self._hash_token(token) % (2**32))
            vec = rng.standard_normal(self._dim, dtype=np.float32)
            self._cache[token] = vec
        return vec

    def _embed_single(self, text: str) -> npt.NDArray[np.float32]:
        """Generate a deterministic pseudo-random vector for a single text.

        Stemmed content words are used so stopwords ("what", "the", "between") do not dominate
        similarity; texts made only of stopwords fall back to their raw tokens.
        """
        tokens = [t for t in content_stems(text) if len(t) > 1] or _TOKEN_RE.findall(text.lower())
        if not tokens:
            return np.full(self._dim, 1.0 / np.sqrt(self._dim), dtype=np.float32)

        vec = np.zeros(self._dim, dtype=np.float32)
        for token in tokens:
            vec += self._token_vector(token)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return np.full(self._dim, 1.0 / np.sqrt(self._dim), dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed multiple texts deterministically."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.vstack([self._embed_single(t) for t in texts]).astype(np.float32)
