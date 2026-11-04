"""Unit tests for retrieval, reciprocal rank fusion (RRF), persistence, and reranking."""

import json
import sys
import types
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
    assert pytest.approx(fused[0].score, rel=1e-5) == (1.0 / 61.0) + (1.0 / 61.0)

    expected_score_bc = (1.0 / 62.0) + (1.0 / 63.0)
    assert pytest.approx(fused[1].score, rel=1e-5) == expected_score_bc
    assert pytest.approx(fused[2].score, rel=1e-5) == expected_score_bc
    # Ties are broken by chunk id so the order is deterministic.
    assert [r.chunk.chunk_id for r in fused[1:]] == ["B", "C"]
    assert fused[0].metadata["r1_rank"] == 1
    assert fused[0].metadata["r2_rank"] == 1


def test_rrf_handles_disjoint_lists_and_empty_input() -> None:
    """A chunk found by only one retriever still gets a score; no lists gives no results."""
    ca = Chunk(chunk_id="A", doc_id="d1", text="a")
    cb = Chunk(chunk_id="B", doc_id="d2", text="b")
    fused = reciprocal_rank_fusion(
        [
            [SearchResult(chunk=ca, score=1.0, rank=1, retriever_name="x")],
            [SearchResult(chunk=cb, score=1.0, rank=2, retriever_name="y")],
        ],
        k=60,
        top_k=5,
    )
    assert [r.chunk.chunk_id for r in fused] == ["A", "B"]
    assert reciprocal_rank_fusion([]) == []


def test_bm25_ranks_exact_term_first() -> None:
    """BM25 puts the chunk sharing rare query terms at rank 1."""
    bm25 = BM25Retriever(chunks=create_sample_chunks())
    results = bm25.retrieve("reciprocal rank fusion", top_k=2)
    assert results[0].chunk.chunk_id == "chunk_4"
    assert results[0].rank == 1 and results[1].rank == 2
    assert bm25.retrieve("???", top_k=2) == []
    assert bm25.retrieve("fusion", top_k=0) == []


def test_bm25_serialization_round_trip() -> None:
    """to_dict/from_dict preserves parameters and chunks."""
    bm25 = BM25Retriever(chunks=create_sample_chunks(), k1=1.2, b=0.5)
    clone = BM25Retriever.from_dict(json.loads(json.dumps(bm25.to_dict())))
    assert (clone.k1, clone.b) == (1.2, 0.5)
    assert [c.chunk_id for c in clone.chunks] == [c.chunk_id for c in bm25.chunks]


def test_dense_retrieval_prefers_overlapping_text() -> None:
    """With the hash embedder, the chunk sharing the most content words ranks first."""
    dense = DenseRetriever(embedder=HashEmbedder(dimension=128), chunks=create_sample_chunks())
    results = dense.retrieve("decision trees boundaries", top_k=1)
    assert results[0].chunk.chunk_id == "chunk_3"
    assert results[0].retriever_name == "dense"
    assert len(dense.to_dict()["chunks"]) == 4


def test_retrieval_determinism() -> None:
    """Hybrid, BM25, and Dense retrieval return identical results on identical queries."""
    hybrid = HybridRetriever(embedder=HashEmbedder(dimension=64), chunks=create_sample_chunks())
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
    embedder = HashEmbedder(dimension=64)
    original_hybrid = HybridRetriever(embedder=embedder, chunks=create_sample_chunks(), rrf_k=50)
    query = "decision trees boundaries"
    original_results = original_hybrid.retrieve(query, top_k=2)

    save_dir = tmp_path / "test_index"
    IndexStore.save(original_hybrid, save_dir)

    assert (save_dir / "metadata.json").exists()
    assert (save_dir / "dense.faiss").exists()
    assert IndexStore.exists(save_dir)

    loaded_hybrid = IndexStore.load(save_dir, embedder=embedder)
    assert len(loaded_hybrid.chunks) == len(original_hybrid.chunks)
    assert loaded_hybrid.rrf_k == 50

    loaded_results = loaded_hybrid.retrieve(query, top_k=2)
    assert len(loaded_results) == len(original_results)
    for r1, r2 in zip(original_results, loaded_results, strict=True):
        assert r1.chunk.chunk_id == r2.chunk.chunk_id
        assert pytest.approx(r1.score) == r2.score


def test_index_load_reembeds_on_embedder_mismatch(tmp_path: Path) -> None:
    """Loading with a different embedder rebuilds the dense index instead of failing."""
    IndexStore.save(
        HybridRetriever(embedder=HashEmbedder(dimension=64), chunks=create_sample_chunks()),
        tmp_path,
    )
    loaded = IndexStore.load(tmp_path, embedder=HashEmbedder(dimension=32))
    assert loaded.dense_retriever.index is not None
    assert loaded.dense_retriever.index.d == 32
    assert loaded.retrieve("gradient descent", top_k=1)[0].chunk.chunk_id == "chunk_1"


def test_index_load_missing_metadata(tmp_path: Path) -> None:
    """Loading from an empty directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        IndexStore.load(tmp_path, embedder=HashEmbedder(dimension=8))


def test_attach_index_validates_shape() -> None:
    """attach_index rejects an index whose size or dimension does not match."""
    chunks = create_sample_chunks()
    source = DenseRetriever(embedder=HashEmbedder(dimension=16), chunks=chunks)
    target = DenseRetriever(embedder=HashEmbedder(dimension=16))
    assert source.index is not None
    with pytest.raises(ValueError, match="holds 4 vectors"):
        target.attach_index(chunks[:2], source.index)
    other_dim = DenseRetriever(embedder=HashEmbedder(dimension=8))
    with pytest.raises(ValueError, match="dimension"):
        other_dim.attach_index(chunks, source.index)


def test_empty_index_retrieval() -> None:
    """Empty retriever instances return empty lists without throwing errors."""
    embedder = HashEmbedder(dimension=64)
    assert BM25Retriever().retrieve("some query") == []
    assert DenseRetriever(embedder=embedder).retrieve("some query") == []
    assert HybridRetriever(embedder=embedder).retrieve("some query") == []
    empty_dense = DenseRetriever(embedder=embedder)
    empty_dense.index_chunks([])
    assert empty_dense.index is None


def test_k_larger_than_corpus() -> None:
    """When top_k is larger than corpus size, return all available chunks without errors."""
    chunks = create_sample_chunks()
    embedder = HashEmbedder(dimension=64)

    assert len(BM25Retriever(chunks=chunks).retrieve("gradient descent", top_k=100)) <= 4
    dense_res = DenseRetriever(embedder=embedder, chunks=chunks).retrieve("gradient", top_k=100)
    hybrid_res = HybridRetriever(embedder=embedder, chunks=chunks).retrieve("gradient", top_k=100)
    assert len(dense_res) == 4
    assert len(hybrid_res) == 4


def test_cross_encoder_reranker_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """CrossEncoderReranker fallback maintains original ordering if model fails to load."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    reranker = CrossEncoderReranker(
        model_name="nonexistent-reranker-model", fallback_pass_through=True
    )
    chunks = create_sample_chunks()
    results = [
        SearchResult(chunk=chunks[0], score=0.8, rank=1, retriever_name="dense"),
        SearchResult(chunk=chunks[1], score=0.6, rank=2, retriever_name="dense"),
    ]
    reranked = reranker.rerank("query", results, top_k=2)
    assert [r.chunk.chunk_id for r in reranked] == ["chunk_1", "chunk_2"]
    assert not reranker.available
    assert reranker.rerank("query", [], top_k=2) == []


def test_cross_encoder_reranker_raises_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """With fallback disabled a missing package surfaces as an error."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    reranker = CrossEncoderReranker(fallback_pass_through=False)
    chunk = create_sample_chunks()[0]
    with pytest.raises(ImportError):
        reranker.rerank("q", [SearchResult(chunk=chunk, score=1.0)], top_k=1)


def test_cross_encoder_reranker_reorders_with_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a (stub) model, results are re-sorted by cross-encoder score with diagnostics."""

    class StubCrossEncoder:
        def __init__(self, name: str, device: str) -> None:
            self.name = name

        def predict(self, pairs: list[list[str]]) -> list[float]:
            return [float(len(text)) for _, text in pairs]

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = StubCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    chunks = create_sample_chunks()
    results = [
        SearchResult(chunk=c, score=1.0 / (i + 1), rank=i + 1, retriever_name="hybrid")
        for i, c in enumerate(chunks)
    ]
    reranker = CrossEncoderReranker()
    assert reranker.available
    reranked = reranker.rerank("q", results, top_k=2)
    assert [r.chunk.chunk_id for r in reranked] == ["chunk_2", "chunk_1"]  # longest texts
    assert reranked[0].rank == 1
    assert reranked[0].metadata["pre_rerank_rank"] == 2
    assert reranked[0].metadata["cross_encoder_score"] == float(len(chunks[1].text))
