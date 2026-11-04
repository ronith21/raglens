"""Okapi BM25 Lexical Retriever."""

from typing import Any

from rank_bm25 import BM25Okapi

from raglens.ingest.models import Chunk
from raglens.retrieve.base import BaseRetriever, SearchResult
from raglens.textutil import tokenize


def default_tokenizer(text: str) -> list[str]:
    """Tokenize with the project-wide tokenizer so BM25, embeddings and gating agree."""
    return tokenize(text)


class BM25Retriever(BaseRetriever):
    """BM25 Lexical retriever based on rank_bm25."""

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] = []
        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Build the BM25 inverted index from a list of chunks."""
        self.chunks = list(chunks)
        self.tokenized_corpus = [default_tokenizer(chunk.text) for chunk in self.chunks]
        self.bm25 = (
            BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
            if self.tokenized_corpus
            else None
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve top_k chunks by BM25 score."""
        if not self.chunks or self.bm25 is None or top_k <= 0:
            return []

        tokens = default_tokenizer(query)
        if not tokens:
            return []

        scores: list[float] = [float(s) for s in self.bm25.get_scores(tokens)]
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = indexed_scores[: min(top_k, len(self.chunks))]

        return [
            SearchResult(
                chunk=self.chunks[idx],
                score=score,
                rank=rank,
                retriever_name="bm25",
                metadata={"bm25_score": score},
            )
            for rank, (idx, score) in enumerate(top_indices, start=1)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for disk persistence."""
        return {
            "k1": self.k1,
            "b": self.b,
            "chunks": [c.model_dump() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BM25Retriever":
        """Deserialize retriever from dictionary."""
        chunks = [Chunk(**c) for c in data.get("chunks", [])]
        return cls(
            chunks=chunks,
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
        )
