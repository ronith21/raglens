"""Tests for acronym mining, query rewriting and multi-query retrieval."""

import pytest

from raglens.embed.hash import HashEmbedder
from raglens.ingest.models import Chunk
from raglens.retrieve.base import BaseRetriever, SearchResult
from raglens.retrieve.hybrid import HybridRetriever
from raglens.retrieve.multi_query import MultiQueryRetriever, QueryRewriter, mine_acronyms

CHUNKS = [
    Chunk(
        chunk_id="rrf",
        doc_id="d1",
        text="Reciprocal Rank Fusion (RRF) fuses rankings. RRF uses a constant k of 60.",
    ),
    Chunk(
        chunk_id="bpe",
        doc_id="d2",
        text="Byte-Pair Encoding (BPE) merges frequent pairs. The (Note) here is not an acronym.",
    ),
    Chunk(
        chunk_id="dvc",
        doc_id="d3",
        text="DVC (Data Version Control) tracks datasets alongside code commits.",
    ),
    Chunk(
        chunk_id="smote",
        doc_id="d4",
        text="Synthetic Minority Over-sampling (SMO) interpolates new minority samples.",
    ),
]


def test_mine_acronyms_both_directions_and_hyphens() -> None:
    """Long-form-first, acronym-first and hyphenated definitions are all found."""
    glossary = mine_acronyms(CHUNKS)
    assert glossary["rrf"] == "reciprocal rank fusion"
    assert glossary["bpe"] == "byte-pair encoding"  # hyphen parts give two initials
    assert glossary["dvc"] == "data version control"  # acronym first
    assert glossary["smo"] == "synthetic minority over-sampling"  # hyphen gives one initial
    assert "note" not in glossary  # initials of the preceding words do not spell it


def test_mine_acronyms_prefers_most_frequent_definition() -> None:
    """When an acronym has competing definitions the most common one wins."""
    chunks = [
        Chunk(chunk_id=str(i), doc_id="d", text=text)
        for i, text in enumerate(
            [
                "Mean Reciprocal Rank (MRR) is a metric.",
                "Mean Reciprocal Rank (MRR) again.",
                "Many Random Robots (MRR) once.",
            ]
        )
    ]
    assert mine_acronyms(chunks)["mrr"] == "mean reciprocal rank"


def test_rewriter_expands_acronyms_and_long_forms() -> None:
    """Acronyms become long forms; long forms get their acronym appended."""
    rewriter = QueryRewriter({"rrf": "reciprocal rank fusion"})
    assert rewriter.expand("Why use RRF?") == "why use reciprocal rank fusion"
    assert (
        rewriter.expand("What is reciprocal rank fusion?") == "what is reciprocal rank fusion rrf"
    )
    assert rewriter.expand("BM25 only") == "bm25 only"


def test_variants_are_distinct_ordered_and_capped() -> None:
    """Original first, then keyword form, then expansion; duplicates removed."""
    rewriter = QueryRewriter({"rrf": "reciprocal rank fusion"})
    assert rewriter.variants("Why use RRF?") == [
        "Why use RRF?",
        "rrf",
        "why use reciprocal rank fusion",
    ]
    assert QueryRewriter(max_variants=1).variants("What is BM25?") == ["What is BM25?"]
    assert QueryRewriter().variants("bm25") == ["bm25"]  # all forms identical
    with pytest.raises(ValueError):
        QueryRewriter(max_variants=0)


class RecordingRetriever(BaseRetriever):
    """Returns a fixed ranking per query and records the queries it saw."""

    def __init__(self, rankings: dict[str, list[Chunk]]) -> None:
        self.rankings = rankings
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.queries.append(query)
        chunks = self.rankings.get(query, [])
        return [SearchResult(chunk=c, score=1.0, rank=i) for i, c in enumerate(chunks, 1)]


def test_multi_query_fuses_variant_rankings() -> None:
    """A chunk ranked well by several variants beats one ranked first by a single variant."""
    a, b, c = CHUNKS[0], CHUNKS[1], CHUNKS[2]
    base = RecordingRetriever(
        {
            "Why use RRF?": [b, a],
            "rrf": [a, c],
            "why use reciprocal rank fusion": [a, b],
        }
    )
    retriever = MultiQueryRetriever(base, QueryRewriter({"rrf": "reciprocal rank fusion"}))
    results = retriever.retrieve("Why use RRF?", top_k=2)

    assert base.queries == ["Why use RRF?", "rrf", "why use reciprocal rank fusion"]
    assert [r.chunk.chunk_id for r in results] == ["rrf", "bpe"]
    assert results[0].retriever_name == "multi-query"
    assert results[0].metadata["query_variants"] == base.queries
    assert results[0].rank == 1
    assert retriever.retrieve("Why use RRF?", top_k=0) == []


def test_multi_query_over_real_hybrid_retriever() -> None:
    """End to end on a hybrid index: the acronym query finds the long-form passage."""
    hybrid = HybridRetriever(embedder=HashEmbedder(dimension=64), chunks=CHUNKS)
    retriever = MultiQueryRetriever(hybrid, QueryRewriter.from_chunks(CHUNKS))
    assert retriever.retrieve("What does DVC track?", top_k=1)[0].chunk.chunk_id == "dvc"
