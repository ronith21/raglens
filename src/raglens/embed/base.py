"""Base interface for embedding models."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class BaseEmbedder(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced embedding vectors."""

    @property
    def name(self) -> str:
        """Human-readable identifier of the embedding backend."""
        return type(self).__name__

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed a list of text strings into an (N, dimension) float32 numpy array.

        Vectors should be L2-normalized so that inner products correspond to cosine similarities.
        """

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single query string into a 1D float32 numpy array of length `dimension`."""
        matrix = self.embed_texts([text])
        return np.asarray(matrix[0], dtype=np.float32)
