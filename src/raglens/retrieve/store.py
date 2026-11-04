"""Index serialization and loading for disk persistence."""

import json
import logging
from pathlib import Path
from typing import Any

import faiss

from raglens.embed.base import BaseEmbedder
from raglens.ingest.models import Chunk
from raglens.retrieve.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

METADATA_FILE = "metadata.json"
FAISS_FILE = "dense.faiss"


class IndexStore:
    """Manages disk persistence and re-loading of hybrid retrieval indices."""

    @staticmethod
    def save(retriever: HybridRetriever, directory: str | Path) -> None:
        """Save BM25 parameters, the FAISS index, and chunks to ``directory``."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "rrf_k": retriever.rrf_k,
            "bm25_k1": retriever.bm25_retriever.k1,
            "bm25_b": retriever.bm25_retriever.b,
            "embedder": retriever.embedder.name,
            "dimension": retriever.embedder.dimension,
            "num_chunks": len(retriever.chunks),
            "chunks": [chunk.model_dump() for chunk in retriever.chunks],
        }
        (dir_path / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if retriever.dense_retriever.index is not None:
            faiss.write_index(retriever.dense_retriever.index, str(dir_path / FAISS_FILE))

    @staticmethod
    def exists(directory: str | Path) -> bool:
        """Return True when a saved index is present in ``directory``."""
        return (Path(directory) / METADATA_FILE).exists()

    @staticmethod
    def load(directory: str | Path, embedder: BaseEmbedder) -> HybridRetriever:
        """Load a hybrid retriever from disk without re-embedding the chunks.

        If the stored FAISS index is missing or was built with a different embedding
        dimension, the dense index is rebuilt from the stored chunks.
        """
        dir_path = Path(directory)
        meta_file = dir_path / METADATA_FILE
        if not meta_file.exists():
            raise FileNotFoundError(f"Index metadata not found at {meta_file}")

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in meta.get("chunks", [])]
        retriever = HybridRetriever(
            embedder=embedder,
            rrf_k=meta.get("rrf_k", 60),
            bm25_k1=meta.get("bm25_k1", 1.5),
            bm25_b=meta.get("bm25_b", 0.75),
        )
        retriever.chunks = chunks
        retriever.bm25_retriever.index_chunks(chunks)

        faiss_file = dir_path / FAISS_FILE
        stored_name = meta.get("embedder")
        if faiss_file.exists() and stored_name in (None, embedder.name):
            try:
                retriever.dense_retriever.attach_index(chunks, faiss.read_index(str(faiss_file)))
                return retriever
            except ValueError as exc:
                logger.warning("Stored FAISS index unusable (%s); re-embedding chunks.", exc)
        elif faiss_file.exists():
            logger.warning(
                "Index was built with embedder %s but %s was given; re-embedding chunks.",
                stored_name,
                embedder.name,
            )
        retriever.dense_retriever.index_chunks(chunks)
        return retriever
