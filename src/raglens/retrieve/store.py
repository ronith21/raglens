"""Index serialization and loading for disk persistence."""

import json
from pathlib import Path
from typing import Any

import faiss

from raglens.embed.base import BaseEmbedder
from raglens.ingest.models import Chunk
from raglens.retrieve.hybrid import HybridRetriever


class IndexStore:
    """Manages disk persistence and re-loading of hybrid retrieval indices."""

    @staticmethod
    def save(
        retriever: HybridRetriever,
        directory: str | Path,
    ) -> None:
        """Save BM25, FAISS index, and chunks to directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "rrf_k": retriever.rrf_k,
            "bm25_k1": retriever.bm25_retriever.k1,
            "bm25_b": retriever.bm25_retriever.b,
            "num_chunks": len(retriever.chunks),
            "chunks": [chunk.model_dump() for chunk in retriever.chunks],
        }
        with open(dir_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if retriever.dense_retriever.index is not None:
            faiss.write_index(retriever.dense_retriever.index, str(dir_path / "dense.faiss"))

    @staticmethod
    def load(
        directory: str | Path,
        embedder: BaseEmbedder,
    ) -> HybridRetriever:
        """Load hybrid retriever from disk directory."""
        dir_path = Path(directory)
        meta_file = dir_path / "metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Index metadata not found at {meta_file}")

        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)

        chunks = [Chunk(**c) for c in meta.get("chunks", [])]
        retriever = HybridRetriever(
            embedder=embedder,
            chunks=chunks,
            rrf_k=meta.get("rrf_k", 60),
            bm25_k1=meta.get("bm25_k1", 1.5),
            bm25_b=meta.get("bm25_b", 0.75),
        )

        faiss_file = dir_path / "dense.faiss"
        if faiss_file.exists():
            index = faiss.read_index(str(faiss_file))
            retriever.dense_retriever.index = index

        return retriever
