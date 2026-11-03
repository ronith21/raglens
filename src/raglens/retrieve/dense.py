"""Dense FAISS vector retriever using IndexFlatIP with normalized vectors."""

from typing import Any

import faiss
import numpy as np

from raglens.embed.base import BaseEmbedder
from raglens.ingest.models import Chunk
from raglens.retrieve.base import BaseRetriever, SearchResult


class DenseRetriever(BaseRetriever):
    """Dense retriever leveraging FAISS IndexFlatIP (cosine similarity on unit vectors)."""

    def __init__(self, embedder: BaseEmbedder, chunks: list[Chunk] | None = None) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.index: faiss.Index | None = None
        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Embed chunks and build the FAISS index."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.index = None
            return

        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.embedder.embed_texts(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        embeddings_c = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings_c)

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve top_k chunks by cosine similarity."""
        if not self.chunks or self.index is None or top_k <= 0:
            return []

        query_vec = self.embedder.embed_query(query)
        query_mat = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)

        actual_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_mat, actual_k)

        results: list[SearchResult] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0], strict=False), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    rank=rank,
                    retriever_name="dense",
                    metadata={"cosine_similarity": float(score)},
                )
            )
        return results

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata and chunk mappings."""
        return {
            "chunks": [c.model_dump() for c in self.chunks],
        }
