"""Base interface for embedding models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt


class BaseEmbedder(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced embedding vectors."""
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed a list of text strings into an (N, dimension) float32 numpy array.

        Vectors should be L2-normalized so that inner products correspond to cosine similarities.
        """
        pass

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single query string into a 1D float32 numpy array of length `dimension`."""
        matrix = self.embed_texts([text])
        return np.asarray(matrix[0], dtype=np.float32)


class FittableEmbedder(BaseEmbedder):
    """An embedder that learns corpus statistics (vocabulary, IDF, topics) before use.

    The dense retriever calls :meth:`fit` on the chunk texts when it builds its index, and the
    index store persists the learned state so queries are embedded in the same space later.
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the embedder has learned its corpus statistics."""

    @abstractmethod
    def fit(self, texts: list[str]) -> "FittableEmbedder":
        """Learn corpus statistics from ``texts``."""

    @abstractmethod
    def save_state(self, directory: str | Path) -> None:
        """Persist the learned state (no pickle)."""

    @abstractmethod
    def load_state(self, directory: str | Path) -> None:
        """Restore state written by :meth:`save_state`."""

    @abstractmethod
    def has_saved_state(self, directory: str | Path) -> bool:
        """Whether ``directory`` contains state for this embedder."""
