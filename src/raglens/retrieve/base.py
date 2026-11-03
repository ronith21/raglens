"""Retrieval models and interfaces."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from raglens.ingest.models import Chunk


class SearchResult(BaseModel):
    """Container for retrieved chunks along with relevance scores and ranking metadata."""

    chunk: Chunk = Field(..., description="The retrieved chunk")
    score: float = Field(..., description="Relevance score (cosine sim, BM25, or RRF)")
    rank: int = Field(1, description="1-indexed position in returned ranking")
    retriever_name: str = Field("", description="Originating retriever (bm25, dense, hybrid)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Diagnostic scores/info")


class BaseRetriever(ABC):
    """Abstract interface for document retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve the top-k most relevant SearchResult items for a query."""
        pass
