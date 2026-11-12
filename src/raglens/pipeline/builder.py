"""Assemble a ready-to-use pipeline from a corpus directory or a saved index."""

import logging
from pathlib import Path

from raglens.chunking.splitter import RecursiveCharacterChunker
from raglens.config import Settings
from raglens.embed.base import BaseEmbedder
from raglens.embed.factory import build_embedder
from raglens.ingest.models import Chunk
from raglens.ingest.parser import ingest_directory
from raglens.llm import build_llm
from raglens.llm.base import BaseLLM
from raglens.pipeline.rag import RAGPipeline
from raglens.rerank.cross_encoder import CrossEncoderReranker
from raglens.retrieve.hybrid import HybridRetriever
from raglens.retrieve.store import IndexStore

logger = logging.getLogger(__name__)


def load_chunks(corpus_dir: str | Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Ingest every supported file under ``corpus_dir`` and split it into chunks."""
    docs = ingest_directory(corpus_dir)
    if not docs:
        raise FileNotFoundError(f"No .md, .txt or .pdf files found in {corpus_dir}")
    chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_documents(docs)


def build_retriever(
    chunks: list[Chunk], embedder: BaseEmbedder, settings: Settings
) -> HybridRetriever:
    """Index chunks into BM25 + FAISS."""
    return HybridRetriever(
        embedder=embedder,
        chunks=chunks,
        rrf_k=settings.rrf_k,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
    )


def build_pipeline(
    settings: Settings,
    llm: BaseLLM | None = None,
    embedder: BaseEmbedder | None = None,
) -> RAGPipeline:
    """Load the saved index if present, otherwise index ``settings.corpus_dir``."""
    embedder = embedder or build_embedder(
        settings.embedder,
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        hash_dim=settings.hash_embed_dim,
        lsa_components=settings.lsa_components,
    )
    index_dir = Path(settings.index_dir)
    if (index_dir / "metadata.json").exists():
        logger.info("Loading index from %s", index_dir)
        retriever = IndexStore.load(index_dir, embedder=embedder)
    else:
        logger.info("No saved index; indexing %s", settings.corpus_dir)
        chunks = load_chunks(settings.corpus_dir, settings.chunk_size, settings.chunk_overlap)
        retriever = build_retriever(chunks, embedder, settings)
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.enable_reranker else None
    return RAGPipeline(
        retriever=retriever,
        llm=llm or build_llm(settings),
        reranker=reranker,
        refusal_threshold=settings.refusal_threshold,
    )
