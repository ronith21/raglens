"""SentenceTransformer embedding model wrapper with lazy loading and CPU defaults."""

import importlib
import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from raglens.embed.base import BaseEmbedder
from raglens.embed.hash import HashEmbedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformers wrapper with lazy-loading and automatic fallback.

    ``sentence-transformers`` is an optional extra (``pip install raglens[models]``). When it is
    missing or the model cannot be downloaded, the embedder falls back to ``HashEmbedder`` and
    logs a warning (unless ``fallback_to_hash`` is False, in which case the error propagates).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        fallback_to_hash: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.fallback_to_hash = fallback_to_hash
        self._model: Any = None
        self._dimension: int | None = None
        self._fallback_embedder: HashEmbedder | None = None

    def _load_model(self) -> None:
        """Lazily load the sentence transformer model."""
        if self._model is not None or self._fallback_embedder is not None:
            return

        try:
            module = importlib.import_module("sentence_transformers")
            self._model = module.SentenceTransformer(self.model_name, device=self.device)
            dummy_vec = self._model.encode(["test"], convert_to_numpy=True)
            self._dimension = int(dummy_vec.shape[1])
            logger.info("Loaded SentenceTransformer %s (dim=%d)", self.model_name, self._dimension)
        except Exception as exc:
            if not self.fallback_to_hash:
                raise
            logger.warning(
                "Failed to load SentenceTransformer %s: %s. Falling back to HashEmbedder.",
                self.model_name,
                exc,
            )
            self._model = None
            self._fallback_embedder = HashEmbedder(dimension=384)
            self._dimension = 384

    @property
    def is_fallback(self) -> bool:
        """True when the hash fallback is in use instead of the real model."""
        self._load_model()
        return self._fallback_embedder is not None

    @property
    def name(self) -> str:
        """Model name, or the fallback's name if the model failed to load."""
        return "hash-384 (fallback)" if self.is_fallback else self.model_name

    @property
    def dimension(self) -> int:
        """Embedding dimension (loads the model on first access)."""
        self._load_model()
        return self._dimension or 384

    def embed_texts(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed texts with the model (or the fallback) as L2-normalised float32 vectors."""
        self._load_model()
        if self._fallback_embedder is not None:
            return self._fallback_embedder.embed_texts(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
