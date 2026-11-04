"""Reciprocal Rank Fusion (RRF) and Hybrid Retriever."""

from typing import Any

from raglens.embed.base import BaseEmbedder
from raglens.ingest.models import Chunk
from raglens.retrieve.base import BaseRetriever, SearchResult
from raglens.retrieve.bm25 import BM25Retriever
from raglens.retrieve.dense import DenseRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    top_k: int = 5,
) -> list[SearchResult]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion: score = sum(1 / (k + rank))."""
    if not ranked_lists:
        return []

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}
    diagnostic_map: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for item in ranked_list:
            cid = item.chunk.chunk_id
            chunk_map[cid] = item.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + item.rank)
            diag = diagnostic_map.setdefault(cid, {})
            retriever_name = item.retriever_name or "unknown"
            diag[f"{retriever_name}_rank"] = item.rank
            diag[f"{retriever_name}_score"] = item.score

    # Sort by fused score, breaking ties by chunk id so results are fully deterministic.
    sorted_items = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))
    results: list[SearchResult] = []
    for rank, (cid, score) in enumerate(sorted_items[: max(top_k, 0)], start=1):
        metadata = dict(diagnostic_map[cid])
        metadata["rrf_score"] = score
        results.append(
            SearchResult(
                chunk=chunk_map[cid],
                score=score,
                rank=rank,
                retriever_name="hybrid",
                metadata=metadata,
            )
        )
    return results


class HybridRetriever(BaseRetriever):
    """Hybrid Retriever combining BM25 and Dense FAISS search via RRF."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        chunks: list[Chunk] | None = None,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self.bm25_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
        self.dense_retriever = DenseRetriever(embedder=embedder)
        if chunks:
            self.index_chunks(chunks)

    @property
    def embedder(self) -> BaseEmbedder:
        """Embedder used by the dense side."""
        return self.dense_retriever.embedder

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks across both BM25 and Dense engines."""
        self.chunks = list(chunks)
        self.bm25_retriever.index_chunks(chunks)
        self.dense_retriever.index_chunks(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve results using Reciprocal Rank Fusion across BM25 and Dense."""
        if not self.chunks or top_k <= 0:
            return []

        candidate_pool_size = max(top_k * 2, 20)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_pool_size)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_pool_size)
        return reciprocal_rank_fusion(
            ranked_lists=[bm25_results, dense_results],
            k=self.rrf_k,
            top_k=top_k,
        )
