"""Index serialization and loading for disk persistence."""

import json
from pathlib import Path
from typing import Any

import faiss

from raglens.embed.base import BaseEmbedder, FittableEmbedder
from raglens.ingest.models import Chunk
from raglens.retrieve.hybrid import HybridRetriever


class IndexStore:
    """Manages disk persistence and re-loading of hybrid retrieval indices."""

    @staticmethod
    def save(
        retriever: HybridRetriever,
        directory: str | Path,
    ) -> None:
        """Save chunks, BM25 parameters, the FAISS index and any fitted embedder state."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        embedder = retriever.dense_retriever.embedder
        meta: dict[str, Any] = {
            "rrf_k": retriever.rrf_k,
            "bm25_k1": retriever.bm25_retriever.k1,
            "bm25_b": retriever.bm25_retriever.b,
            "embedder": type(embedder).__name__,
            "num_chunks": len(retriever.chunks),
            "chunks": [chunk.model_dump() for chunk in retriever.chunks],
        }
        with open(dir_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if retriever.dense_retriever.index is not None:
            faiss.write_index(retriever.dense_retriever.index, str(dir_path / "dense.faiss"))
        if isinstance(embedder, FittableEmbedder) and embedder.is_fitted:
            embedder.save_state(dir_path)

    @staticmethod
    def load(
        directory: str | Path,
        embedder: BaseEmbedder,
    ) -> HybridRetriever:
        """Load a hybrid retriever from disk without re-embedding the corpus.

        If the index was built with a :class:`FittableEmbedder` (LSA, word vectors), its saved
        corpus statistics are restored so queries are embedded exactly like the chunks were.
        """
        dir_path = Path(directory)
        meta_file = dir_path / "metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Index metadata not found at {meta_file}")

        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)

        if isinstance(embedder, FittableEmbedder) and embedder.has_saved_state(dir_path):
            embedder.load_state(dir_path)

        chunks = [Chunk(**c) for c in meta.get("chunks", [])]
        retriever = HybridRetriever(
            embedder=embedder,
            rrf_k=meta.get("rrf_k", 60),
            bm25_k1=meta.get("bm25_k1", 1.5),
            bm25_b=meta.get("bm25_b", 0.75),
        )
        faiss_file = dir_path / "dense.faiss"
        if faiss_file.exists() and chunks:
            retriever.chunks = list(chunks)
            retriever.bm25_retriever.index_chunks(chunks)
            retriever.dense_retriever.attach_index(chunks, faiss.read_index(str(faiss_file)))
        elif chunks:
            retriever.index_chunks(chunks)
        return retriever
