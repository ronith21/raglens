"""SentenceTransformer embedding model wrapper with lazy loading and CPU defaults."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from raglens.embed.base import BaseEmbedder
from raglens.embed.hash import HashEmbedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformers wrapper with lazy-loading and automatic fallback."""

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
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            dummy_vec = self._model.encode(["test"], convert_to_numpy=True)
            self._dimension = int(dummy_vec.shape[1])
            logger.info("Loaded SentenceTransformer %s (dim=%d)", self.model_name, self._dimension)
        except Exception as e:
            if self.fallback_to_hash:
                logger.warning(
                    "Failed to load SentenceTransformer %s: %s. Falling back to HashEmbedder.",
                    self.model_name,
                    e,
                )
                self._fallback_embedder = HashEmbedder(dimension=384)
                self._dimension = 384
            else:
                raise e

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    def embed_texts(self, texts: list[str]) -> npt.NDArray[np.float32]:
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
