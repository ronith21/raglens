"""Cross-encoder reranking module with lazy loading and fallback."""

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
        pass


class CrossEncoderReranker(BaseReranker):
    """Lazy-loaded cross-encoder reranker using sentence-transformers CrossEncoder."""

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

    def _load_model(self) -> None:
        """Lazily load cross-encoder model."""
        if self._model is not None or self._load_failed:
            return

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Loaded CrossEncoder %s", self.model_name)
        except Exception as e:
            if self.fallback_pass_through:
                logger.warning(
                    "Failed to load CrossEncoder %s: %s. Using pass-through fallback.",
                    self.model_name,
                    e,
                )
                self._load_failed = True
            else:
                raise e

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        """Rerank input results using the cross-encoder."""
        if not results or top_k <= 0:
            return []

        self._load_model()
        if self._load_failed or self._model is None:
            return results[: min(top_k, len(results))]

        pairs = [[query, r.chunk.text] for r in results]
        scores = self._model.predict(pairs)

        scored_pairs = list(zip(results, scores, strict=False))
        scored_pairs.sort(key=lambda x: float(x[1]), reverse=True)

        reranked_results: list[SearchResult] = []
        for rank, (orig_res, score) in enumerate(scored_pairs[:top_k], start=1):
            metadata = dict(orig_res.metadata)
            metadata["pre_rerank_rank"] = orig_res.rank
            metadata["pre_rerank_score"] = orig_res.score
            metadata["cross_encoder_score"] = float(score)

            reranked_results.append(
                SearchResult(
                    chunk=orig_res.chunk,
                    score=float(score),
                    rank=rank,
                    retriever_name=orig_res.retriever_name,
                    metadata=metadata,
                )
            )

        return reranked_results
