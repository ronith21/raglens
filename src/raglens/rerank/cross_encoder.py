"""Cross-encoder reranking module with lazy loading and fallback."""

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any

from raglens.retrieve.base import SearchResult

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface for passage rerankers."""

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        """Rerank a list of SearchResults for a query."""


class CrossEncoderReranker(BaseReranker):
    """Lazy-loaded cross-encoder reranker using sentence-transformers CrossEncoder.

    With ``fallback_pass_through=True`` a missing package or failed model download degrades to
    returning the first-stage ranking unchanged (truncated to ``top_k``).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        fallback_pass_through: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.fallback_pass_through = fallback_pass_through
        self._model: Any = None
        self._load_failed: bool = False

    @property
    def available(self) -> bool:
        """True when the real cross-encoder model is loaded."""
        self._load_model()
        return self._model is not None

    def _load_model(self) -> None:
        """Lazily load cross-encoder model."""
        if self._model is not None or self._load_failed:
            return
        try:
            module = importlib.import_module("sentence_transformers")
            self._model = module.CrossEncoder(self.model_name, device=self.device)
            logger.info("Loaded CrossEncoder %s", self.model_name)
        except Exception as exc:
            if not self.fallback_pass_through:
                raise
            logger.warning(
                "Failed to load CrossEncoder %s: %s. Using pass-through fallback.",
                self.model_name,
                exc,
            )
            self._load_failed = True

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        """Rerank input results using the cross-encoder."""
        if not results or top_k <= 0:
            return []

        self._load_model()
        if self._model is None:
            return results[:top_k]

        scores = self._model.predict([[query, r.chunk.text] for r in results])
        scored_pairs = sorted(
            zip(results, (float(s) for s in scores), strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked: list[SearchResult] = []
        for rank, (orig, score) in enumerate(scored_pairs[:top_k], start=1):
            metadata = dict(orig.metadata)
            metadata["pre_rerank_rank"] = orig.rank
            metadata["pre_rerank_score"] = orig.score
            metadata["cross_encoder_score"] = score
            reranked.append(
                SearchResult(
                    chunk=orig.chunk,
                    score=score,
                    rank=rank,
                    retriever_name=orig.retriever_name,
                    metadata=metadata,
                )
            )
        return reranked
