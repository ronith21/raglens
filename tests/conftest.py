"""Shared fixtures: a pipeline over the real sample corpus using offline components only."""

from pathlib import Path

import pytest

from raglens.config import Settings
from raglens.embed.lsa import LSAEmbedder
from raglens.llm.fake import FakeLLM
from raglens.pipeline.builder import build_retriever, load_chunks
from raglens.pipeline.rag import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "sample_corpus"
QA = ROOT / "data" / "eval" / "qa.jsonl"


@pytest.fixture(scope="session")
def corpus_chunks() -> list:  # type: ignore[type-arg]
    """Sample corpus chunked at 512 characters."""
    return load_chunks(CORPUS, 512, 64)


@pytest.fixture()
def pipeline(corpus_chunks: list) -> RAGPipeline:  # type: ignore[type-arg]
    """Hybrid pipeline with LSA embeddings and the extractive FakeLLM."""
    settings = Settings(corpus_dir=str(CORPUS))
    retriever = build_retriever(corpus_chunks, LSAEmbedder(n_components=64), settings)
    return RAGPipeline(retriever, FakeLLM(), refusal_threshold=0.35)
