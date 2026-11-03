"""Unit tests for retrieval, reciprocal rank fusion (RRF), persistence, and reranking."""

from pathlib import Path

import pytest

from raglens.embed.hash import HashEmbedder
from raglens.ingest.models import Chunk
from raglens.rerank.cross_encoder import CrossEncoderReranker
from raglens.retrieve.base import SearchResult
from raglens.retrieve.bm25 import BM25Retriever
from raglens.retrieve.dense import DenseRetriever
from raglens.retrieve.hybrid import HybridRetriever, reciprocal_rank_fusion
from raglens.retrieve.store import IndexStore


def create_sample_chunks() -> list[Chunk]:
    """Create sample chunks for retrieval testing."""
    return [
        Chunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            text="Gradient descent optimizes parameters iteratively using the loss gradient.",
            page_number=1,
            start_char=0,
            end_char=75,
        ),
        Chunk(
            chunk_id="chunk_2",
            doc_id="doc_2",
            text="Transformers rely on multi-head self-attention mechanisms without recurrence.",
            page_number=1,
            start_char=0,
            end_char=77,
        ),
        Chunk(
            chunk_id="chunk_3",
            doc_id="doc_3",
            text="Decision trees partition features into orthogonal decision boundaries.",
            page_number=1,
            start_char=0,
            end_char=71,
        ),
        Chunk(
            chunk_id="chunk_4",
            doc_id="doc_4",
            text="Reciprocal rank fusion combines rankings without score calibration.",
            page_number=1,
            start_char=0,
            end_char=67,
        ),
    ]


def test_reciprocal_rank_fusion_hand_computed_example() -> None:
    """Verify RRF computation on a hand-calculated mathematical example."""
    ca = Chunk(chunk_id="A", doc_id="d1", text="doc A")
    cb = Chunk(chunk_id="B", doc_id="d2", text="doc B")
    cc = Chunk(chunk_id="C", doc_id="d3", text="doc C")

    list1 = [
        SearchResult(chunk=ca, score=10.0, rank=1, retriever_name="r1"),
        SearchResult(chunk=cb, score=8.0, rank=2, retriever_name="r1"),
        SearchResult(chunk=cc, score=5.0, rank=3, retriever_name="r1"),
    ]
    list2 = [
        SearchResult(chunk=ca, score=0.9, rank=1, retriever_name="r2"),
        SearchResult(chunk=cc, score=0.7, rank=2, retriever_name="r2"),
        SearchResult(chunk=cb, score=0.4, rank=3, retriever_name="r2"),
    ]

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=3)

    assert len(fused) == 3
    assert fused[0].chunk.chunk_id == "A"
    expected_score_a = (1.0 / 61.0) + (1.0 / 61.0)
    assert pytest.approx(fused[0].score, rel=1e-5) == expected_score_a

    expected_score_bc = (1.0 / 62.0) + (1.0 / 63.0)
    assert pytest.approx(fused[1].score, rel=1e-5) == expected_score_bc
    assert pytest.approx(fused[2].score, rel=1e-5) == expected_score_bc


def test_retrieval_determinism() -> None:
    """Hybrid, BM25, and Dense retrieval return identical results on identical queries."""
    chunks = create_sample_chunks()
    embedder = HashEmbedder(dimension=64)

    hybrid = HybridRetriever(embedder=embedder, chunks=chunks, rrf_k=60)
    query = "self-attention mechanism"

    res1 = hybrid.retrieve(query, top_k=3)
    res2 = hybrid.retrieve(query, top_k=3)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2, strict=True):
        assert r1.chunk.chunk_id == r2.chunk.chunk_id
        assert pytest.approx(r1.score) == r2.score
        assert r1.rank == r2.rank


def test_index_save_load_round_trip(tmp_path: Path) -> None:
    """IndexStore correctly saves and reloads BM25 and FAISS dense indices."""
    chunks = create_sample_chunks()
    embedder = HashEmbedder(dimension=64)

    original_hybrid = HybridRetriever(embedder=embedder, chunks=chunks, rrf_k=50)
    query = "decision trees boundaries"
    original_results = original_hybrid.retrieve(query, top_k=2)

    save_dir = tmp_path / "test_index"
    IndexStore.save(original_hybrid, save_dir)

    assert (save_dir / "metadata.json").exists()
    assert (save_dir / "dense.faiss").exists()

    loaded_hybrid = IndexStore.load(save_dir, embedder=embedder)
    assert len(loaded_hybrid.chunks) == len(original_hybrid.chunks)
    assert loaded_hybrid.rrf_k == 50

    loaded_results = loaded_hybrid.retrieve(query, top_k=2)
    assert len(loaded_results) == len(original_results)
    for r1, r2 in zip(original_results, loaded_results, strict=True):
        assert r1.chunk.chunk_id == r2.chunk.chunk_id
        assert pytest.approx(r1.score) == r2.score


def test_empty_index_retrieval() -> None:
    """Empty retriever instances return empty lists without throwing errors."""
    embedder = HashEmbedder(dimension=64)
    bm25 = BM25Retriever()
    dense = DenseRetriever(embedder=embedder)
    hybrid = HybridRetriever(embedder=embedder)

    assert bm25.retrieve("some query") == []
    assert dense.retrieve("some query") == []
    assert hybrid.retrieve("some query") == []


def test_k_larger_than_corpus() -> None:
    """When top_k is larger than corpus size, return all available chunks without errors."""
    chunks = create_sample_chunks()
    embedder = HashEmbedder(dimension=64)

    bm25 = BM25Retriever(chunks=chunks)
    dense = DenseRetriever(embedder=embedder, chunks=chunks)
    hybrid = HybridRetriever(embedder=embedder, chunks=chunks)

    bm25_res = bm25.retrieve("gradient descent", top_k=100)
    dense_res = dense.retrieve("gradient descent", top_k=100)
    hybrid_res = hybrid.retrieve("gradient descent", top_k=100)

    assert len(bm25_res) <= 4
    assert len(dense_res) == 4
    assert len(hybrid_res) == 4


def test_cross_encoder_reranker_fallback() -> None:
    """CrossEncoderReranker fallback maintains original ordering if model fails to load."""
    reranker = CrossEncoderReranker(
        model_name="nonexistent-reranker-model",
        fallback_pass_through=True,
    )
    chunks = create_sample_chunks()
    results = [
        SearchResult(chunk=chunks[0], score=0.8, rank=1, retriever_name="dense"),
        SearchResult(chunk=chunks[1], score=0.6, rank=2, retriever_name="dense"),
    ]
    reranked = reranker.rerank("query", results, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk.chunk_id == "chunk_1"
    assert reranked[1].chunk.chunk_id == "chunk_2"
