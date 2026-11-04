"""Tests for evaluation metrics, dataset I/O, the runner, synthetic QA and the benchmark."""

import json
import math
from pathlib import Path

import pytest

from raglens.engine import RAGEngine
from raglens.eval.benchmark import main as benchmark_main
from raglens.eval.dataset import QAExample, load_dataset, save_dataset
from raglens.eval.metrics import (
    citation_precision,
    dedupe_preserving_order,
    groundedness,
    hit_at_k,
    mean,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    reference_recall,
    token_f1,
)
from raglens.eval.runner import evaluate
from raglens.eval.synthetic import generate_synthetic_qa
from tests.conftest import EVAL_DATASET

RANKED = ["a", "b", "c", "d"]


def test_hit_and_recall() -> None:
    """Hand-checked Hit@k and Recall@k values."""
    assert hit_at_k(RANKED, {"c"}, 2) == 0.0
    assert hit_at_k(RANKED, {"c"}, 3) == 1.0
    assert recall_at_k(RANKED, {"b", "z"}, 4) == 0.5
    assert recall_at_k(RANKED, set(), 4) == 0.0


def test_reciprocal_rank() -> None:
    """RR is 1/position of the first hit, 0 outside the window."""
    assert reciprocal_rank(RANKED, {"c", "d"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"d"}, k=3) == 0.0


def test_ndcg_hand_computed() -> None:
    """Relevant docs at ranks 2 and 4 with two relevant overall."""
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, {"b", "d"}, 4) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(RANKED, {"a"}, 1) == 1.0
    assert ndcg_at_k(RANKED, set(), 3) == 0.0


def test_answer_metrics() -> None:
    """Citation precision, groundedness and helpers."""
    assert citation_precision(["a", "b", "a"], {"a"}) == pytest.approx(2 / 3)
    assert citation_precision([], {"a"}) is None
    assert (
        groundedness(
            "Gradient descent minimizes loss [1].", ["Gradient descent minimizes the loss."]
        )
        == 1.0
    )
    assert groundedness("Paris is lovely [1].", ["Gradient descent"]) == 0.0
    assert groundedness("the of", ["x"]) is None
    assert dedupe_preserving_order(["a", "b", "a", "c"]) == ["a", "b", "c"]
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([], 95) == 0.0
    assert mean([]) == 0.0


def test_dataset_round_trip_and_validation(tmp_path: Path) -> None:
    """JSONL saves and loads; missing ids are filled; bad JSON names the line."""
    path = tmp_path / "qa.jsonl"
    save_dataset([QAExample(id="x", question="Q?", relevant_doc_ids=["d"])], path)
    path.write_text(path.read_text() + '\n{"question": "U?"}\n', encoding="utf-8")
    loaded = load_dataset(path)
    assert [e.id for e in loaded] == ["x", "q3"]
    assert loaded[0].answerable and not loaded[1].answerable

    (tmp_path / "bad.jsonl").write_text("{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_dataset(tmp_path / "bad.jsonl")


def test_shipped_dataset_meets_spec() -> None:
    """The hand-written set has 40+ answerable and 5+ unanswerable questions with valid slugs."""
    examples = load_dataset(EVAL_DATASET)
    answerable = [e for e in examples if e.answerable]
    assert len(answerable) >= 40
    assert len(examples) - len(answerable) >= 5
    corpus = {p.stem for p in (EVAL_DATASET.parents[1] / "sample_corpus").glob("*.md")}
    for example in answerable:
        assert set(example.relevant_doc_ids) <= corpus, example.id
    assert len({e.id for e in examples}) == len(examples)


def test_evaluate_on_sample_corpus(engine: RAGEngine) -> None:
    """End-to-end evaluation produces sane metrics and per-question rows."""
    dataset = load_dataset(EVAL_DATASET)
    report = evaluate(engine.pipeline("hybrid"), dataset, ks=(1, 5), config_name="hybrid@512")

    assert report.config == "hybrid@512"
    assert report.num_questions == len(dataset)
    assert set(report.retrieval) == {
        "hit@1",
        "recall@1",
        "ndcg@1",
        "hit@5",
        "recall@5",
        "ndcg@5",
        "mrr",
    }
    assert report.retrieval["hit@5"] >= 0.9
    assert report.answer["refusal_accuracy"] >= 0.8
    assert 0.0 <= report.answer["citation_precision"] <= 1.0
    assert "retrieve_p95" in report.latency_ms
    assert len(report.questions) == len(dataset)


def test_synthetic_generation_is_reproducible(engine: RAGEngine) -> None:
    """Same seed, same questions; every question points at its source document."""
    first = generate_synthetic_qa(engine.chunks, num_questions=10, seed=7)
    second = generate_synthetic_qa(engine.chunks, num_questions=10, seed=7)
    assert [q.question for q in first] == [q.question for q in second]
    assert len(first) == 10
    assert all(q.synthetic and q.question.endswith("?") and q.relevant_doc_ids for q in first)
    assert generate_synthetic_qa([], 5) == []
    assert generate_synthetic_qa(engine.chunks, 0) == []


def test_benchmark_writes_report_and_charts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ablation CLI writes markdown, JSON and PNGs from a real (small) run."""
    monkeypatch.setenv("USE_HASH_EMBEDDER", "true")
    monkeypatch.setenv("CORPUS_DIR", str(EVAL_DATASET.parents[1] / "sample_corpus"))
    small = tmp_path / "small.jsonl"
    rows = load_dataset(EVAL_DATASET)
    save_dataset(rows[:6] + rows[-2:], small)

    out = tmp_path / "proof"
    assert (
        benchmark_main(["--dataset", str(small), "--out", str(out), "--chunk-sizes", "256,512"])
        == 0
    )

    report = (out / "eval_report.md").read_text(encoding="utf-8")
    assert "| hybrid@512 |" in report
    assert "HashEmbedder" in report  # the limitation note is present for the fallback
    data = json.loads((out / "eval_results.json").read_text(encoding="utf-8"))
    assert len(data["reports"]) == 10  # 5 configs x 2 chunk sizes
    for name in ("retrieval_metrics_512.png", "answer_metrics_512.png", "mrr_heatmap.png"):
        assert (out / name).stat().st_size > 5_000


def test_token_f1_and_reference_recall_hand_computed() -> None:
    """F1 over stemmed content words, with multiset overlap; recall over distinct stems."""
    # prediction stems: learn, rate, large, overshoot ; reference: rate, large, diverge
    prediction = "The learning rate is large, so it overshoots [1]."
    reference = "A large rate diverges."
    precision, recall = 2 / 4, 2 / 3
    assert token_f1(prediction, reference) == pytest.approx(
        2 * precision * recall / (precision + recall)
    )
    assert reference_recall(prediction, reference) == pytest.approx(2 / 3)
    assert token_f1("", reference) == 0.0
    assert token_f1(reference, reference) == pytest.approx(1.0)
    assert reference_recall("anything", "the of") is None
    assert token_f1("rate rate", "rate") == pytest.approx(2 * 0.5 * 1.0 / 1.5)  # multiset


def test_refusal_accuracy_hand_computed(engine: RAGEngine) -> None:
    """Refusal accuracy = refused unanswerable / unanswerable; false refusals likewise."""
    dataset = [
        QAExample(
            id="a1",
            question="How do isolation forests detect anomalies?",
            relevant_doc_ids=["15_anomaly_detection"],
            answer="Outliers need fewer splits.",
        ),
        QAExample(id="u1", question="What is the capital of Australia?"),
        QAExample(id="u2", question="Who won the 2018 FIFA World Cup?"),
    ]
    report = evaluate(engine.pipeline("hybrid"), dataset, ks=(1,))
    refused = {q.id: q.refused for q in report.questions}
    assert refused == {"a1": False, "u1": True, "u2": True}
    assert report.answer["refusal_accuracy"] == 1.0
    assert report.answer["false_refusal_rate"] == 0.0
    assert report.answer["citation_support"] == pytest.approx(1.0)
    assert 0.0 < report.answer["answer_f1"] <= 1.0
    a1 = next(q for q in report.questions if q.id == "a1")
    assert a1.citations_total >= 1 and a1.citations_supported == a1.citations_total


@pytest.mark.parametrize("name", ["banking_dev.jsonl", "banking_test.jsonl"])
def test_banking_datasets_are_valid_and_disjoint(name: str) -> None:
    """Banking sets reference real documents, cover every kind, and do not overlap."""
    path = EVAL_DATASET.parent / name
    examples = load_dataset(path)
    corpus = {p.stem for p in (EVAL_DATASET.parents[1] / "corpora" / "banking").glob("*.md")}
    assert len(corpus) == 14
    assert {e.kind for e in examples} == {"direct", "paraphrase", "multi", "unanswerable"}
    for example in examples:
        assert set(example.relevant_doc_ids) <= corpus, example.id
        assert example.answerable == (example.kind != "unanswerable")
    dev = {e.question for e in load_dataset(EVAL_DATASET.parent / "banking_dev.jsonl")}
    test = {e.question for e in load_dataset(EVAL_DATASET.parent / "banking_test.jsonl")}
    assert not dev & test


def test_report_breaks_down_by_kind(engine: RAGEngine) -> None:
    """by_kind holds counts and refusal shares per question type."""
    dataset = [
        QAExample(
            id="a",
            question="How do isolation forests detect anomalies?",
            relevant_doc_ids=["15_anomaly_detection"],
            kind="paraphrase",
        ),
        QAExample(id="u", question="What is the capital of Australia?"),
    ]
    report = evaluate(engine.pipeline("hybrid"), dataset, ks=(1, 5))
    assert report.by_kind["paraphrase"]["n"] == 1.0
    assert report.by_kind["paraphrase"]["hit@5"] == 1.0
    assert report.by_kind["unanswerable"]["refused"] == 1.0


def test_benchmark_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--corpus/--configs/--gate select what runs; bad configs and a dead LLM exit 1."""
    monkeypatch.setenv("USE_HASH_EMBEDDER", "true")
    root = EVAL_DATASET.parents[1]
    small = tmp_path / "small.jsonl"
    rows = load_dataset(EVAL_DATASET.parent / "banking_dev.jsonl")
    save_dataset(rows[:4] + rows[-2:], small)
    out = tmp_path / "out"
    args = [
        "--dataset",
        str(small),
        "--out",
        str(out),
        "--chunk-sizes",
        "512",
        "--corpus",
        str(root / "corpora" / "banking"),
        "--configs",
        "bm25,hybrid",
        "--gate",
        "rule",
    ]
    assert benchmark_main(args) == 0
    data = json.loads((out / "eval_results.json").read_text(encoding="utf-8"))
    assert [r["config"] for r in data["reports"]] == ["bm25@512", "hybrid@512"]
    assert data["meta"]["gate"] == "RuleGate"
    assert "By question type" in (out / "eval_report.md").read_text(encoding="utf-8")
    assert benchmark_main(["--configs", "nope", "--out", str(out)]) == 1
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    assert benchmark_main([*args, "--llm", "ollama"]) == 1
