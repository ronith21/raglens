"""Evaluation tests: hand-computed metrics, dataset loading, threshold tuning, full run, report."""

import json
from pathlib import Path

import pytest

from raglens.config import Settings
from raglens.eval.dataset import QAItem, load_dataset
from raglens.eval.metrics import (
    citation_precision,
    doc_ranking,
    groundedness,
    hit_at_k,
    ndcg_at_k,
    paired_bootstrap_ci,
    percentile,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)
from raglens.eval.report import write_report
from raglens.eval.runner import EvalConfig, run_evaluation, tune_refusal_threshold
from raglens.eval.synthetic import generate_synthetic
from raglens.ingest.models import Chunk
from raglens.pipeline.rag import RAGPipeline
from raglens.retrieve.base import SearchResult
from raglens.textutil import content_tokens

from .conftest import CORPUS, QA

RANKING = ["a", "b", "c", "d"]


def test_rank_metrics_hand_computed() -> None:
    """Relevant docs {b, d}: first hit at rank 2."""
    rel = {"b", "d"}
    assert hit_at_k(RANKING, rel, 1) == 0.0
    assert hit_at_k(RANKING, rel, 2) == 1.0
    assert recall_at_k(RANKING, rel, 2) == 0.5
    assert recall_at_k(RANKING, rel, 4) == 1.0
    assert recall_at_k(RANKING, set(), 4) == 0.0
    assert reciprocal_rank(RANKING, rel) == 0.5
    assert reciprocal_rank(RANKING, {"z"}) == 0.0
    # DCG = 1/log2(3) + 1/log2(5); ideal = 1 + 1/log2(3)
    expected = (1 / 1.5849625 + 1 / 2.3219281) / (1 + 1 / 1.5849625)
    assert ndcg_at_k(RANKING, rel, 4) == pytest.approx(expected, rel=1e-6)
    assert ndcg_at_k(["b", "d"], rel, 2) == pytest.approx(1.0)
    assert ndcg_at_k(RANKING, set(), 3) == 0.0


def test_doc_ranking_collapses_chunks() -> None:
    """Two chunks from one document count as one entry at the first position."""
    results = [
        SearchResult(chunk=Chunk(chunk_id=c, doc_id=c, text="", metadata={"source": s}), score=0)
        for c, s in [("1", "x"), ("2", "x"), ("3", "y")]
    ]
    assert doc_ranking(results) == ["x", "y"]


def test_answer_metrics() -> None:
    """Citation precision, groundedness and token F1 on small examples."""
    assert citation_precision("a [1] b [2]", ["x", "y"], {"x"}) == 0.5
    assert citation_precision("no citations", [], {"x"}) is None
    assert (
        groundedness("Gradient descent minimises loss [1]", ["gradient descent minimises the loss"])
        == 1.0
    )
    assert groundedness("Gradient descent uses magic [1]", ["gradient descent"]) == pytest.approx(
        2 / 3
    )
    assert groundedness("", ["x"]) is None
    # "uses" is a stopword: 2 of 3 content tokens overlap on each side -> P = R = F1 = 2/3.
    assert token_f1("rank fusion uses ranks", "fusion uses rank positions") == pytest.approx(2 / 3)
    assert token_f1("", "x") == 0.0
    assert token_f1("apples", "oranges") == 0.0


def test_percentile_and_bootstrap() -> None:
    """Percentiles interpolate; the bootstrap CI brackets a clear difference."""
    assert percentile([], 50) == 0.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    mean, lo, hi = paired_bootstrap_ci([1.0] * 30, [0.0] * 15 + [1.0] * 15)
    assert mean == 0.5 and 0 < lo <= mean <= hi
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [])


def test_dataset_file_is_well_formed() -> None:
    """The shipped set has 40+ answerable, 5+ unanswerable rows, and every label is a real file."""
    items = load_dataset(QA)
    sources = {p.stem for p in CORPUS.glob("*.md")}
    answerable = [i for i in items if i.answerable]
    assert len(answerable) >= 40
    assert len([i for i in items if not i.answerable]) >= 5
    assert {i.split for i in items} == {"dev", "test"}
    assert len({i.id for i in items}) == len(items)
    for item in answerable:
        assert set(item.relevant_doc_ids) <= sources, item.id
    assert all(i.split == "dev" for i in load_dataset(QA, split="dev"))


def test_dataset_errors(tmp_path: Path) -> None:
    """Broken rows report their line number; an empty selection is an error."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "1", "question": "q"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_dataset(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No rows"):
        load_dataset(empty)


def test_synthetic_questions_point_at_their_source(corpus_chunks: list[Chunk]) -> None:
    """Each synthetic question is a sentence from a chunk, labelled with that chunk's file."""
    items = generate_synthetic(corpus_chunks, n=10)
    assert 0 < len(items) <= 10
    for item in items:
        assert item.type == "synthetic"
        words = set(content_tokens(item.question))
        source_chunks = [
            c for c in corpus_chunks if c.metadata["source"] == item.relevant_doc_ids[0]
        ]
        assert any(words <= set(content_tokens(c.text)) for c in source_chunks)


def test_threshold_tuning_separates_answerable(pipeline: RAGPipeline) -> None:
    """With clear examples, the tuned threshold answers the real question and refuses the other."""
    items = [
        QAItem(
            id="a",
            question="How does BM25 inverse document frequency work?",
            relevant_doc_ids=["11_lexical_search_bm25"],
        ),
        QAItem(id="b", question="What is a good sourdough recipe?"),
    ]
    threshold, sweep = tune_refusal_threshold(pipeline, items, "hybrid", 5)
    assert len(sweep) == 21
    best = max(r["balanced_accuracy"] for r in sweep)
    assert best == 1.0
    assert next(r for r in sweep if r["threshold"] == threshold)["balanced_accuracy"] == 1.0


def test_full_evaluation_and_report(tmp_path: Path) -> None:
    """A small but complete run produces every section, chart and the JSON dump."""
    items = load_dataset(QA)
    config = EvalConfig(
        embedders=["lsa", "hash"],
        chunk_sizes=[512, 1024],
        main_embedder="lsa",
        main_chunk_size=512,
        synthetic_n=8,
    )
    results = run_evaluation(items, Settings(corpus_dir=str(CORPUS)), config)
    main = results["main"]
    assert set(main["retrieval"]) >= {"bm25", "dense", "hybrid"}
    assert 0.0 <= main["refusal_threshold"] <= 1.0
    assert len(results["ablation"]) == 2 * 2 * len(main["retrieval"])
    assert "hybrid_vs_bm25_mrr" in main["significance"]
    # Answer metrics are computed on the test split only.
    assert main["answers"]["n"] == len([i for i in items if i.split == "test"])

    report = write_report(results, tmp_path)
    text = report.read_text(encoding="utf-8")
    for heading in (
        "Retrieval quality",
        "paired bootstrap",
        "test split only",
        "Ablation",
        "disagree",
    ):
        assert heading.lower() in text.lower()
    for chart in ("mrr_by_type.png", "ablation_ndcg.png", "refusal_threshold.png", "latency.png"):
        assert (tmp_path / chart).stat().st_size > 1000
    assert (
        json.loads((tmp_path / "eval_results.json").read_text())["config"]["main_embedder"] == "lsa"
    )
