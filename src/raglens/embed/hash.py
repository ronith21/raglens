"""Deterministic, dependency-free hash embedder for testing and offline fallbacks."""

import hashlib

import numpy as np
import numpy.typing as npt

from raglens.embed.base import BaseEmbedder


class HashEmbedder(BaseEmbedder):
    """Deterministic hash-based embedder producing normalized pseudo-random vectors."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_token(self, token: str) -> int:
        """Hash a single token string to an integer seed using SHA256."""
        h = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(h[:8], byteorder="big", signed=False)

    def _embed_single(self, text: str) -> npt.NDArray[np.float32]:
        """Generate a deterministic pseudo-random vector for a single text."""
        tokens = text.lower().strip().split()
        if not tokens:
            vec = np.ones(self._dim, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            return (vec / (norm if norm > 0 else 1.0)).astype(np.float32)

        vec = np.zeros(self._dim, dtype=np.float32)
        for token in tokens:
            seed = self._hash_token(token) % (2**32)
            rng = np.random.default_rng(seed)
            token_vec = rng.standard_normal(self._dim, dtype=np.float32)
            vec += token_vec

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        else:
            vec = np.ones(self._dim, dtype=np.float32) / np.sqrt(self._dim)
        return vec.astype(np.float32)

    def embed_texts(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed multiple texts deterministically."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        rows = [self._embed_single(t) for t in texts]
        return np.vstack(rows).astype(np.float32)
