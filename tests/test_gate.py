"""Tests for refusal-gate features, the rule and learned gates, training and gate selection."""

from pathlib import Path

import numpy as np
import pytest

from raglens.config import Settings
from raglens.embed.hash import HashEmbedder
from raglens.engine import load_gate
from raglens.eval.dataset import QAExample
from raglens.eval.train_gate import (
    Sample,
    choose_threshold,
    expected_cost,
    gate_metrics,
    train,
)
from raglens.eval.train_gate import (
    main as train_main,
)
from raglens.ingest.models import Chunk
from raglens.pipeline.gate import (
    FEATURE_NAMES,
    LearnedGate,
    LearnedGateModel,
    RuleGate,
    compute_features,
    specific_terms,
)
from raglens.retrieve.base import SearchResult

CTX = [
    SearchResult(
        chunk=Chunk(
            chunk_id="c1",
            doc_id="d1",
            text="The leverage ratio is Tier 1 capital divided by total exposure. "
            "The minimum is 3%.",
            metadata={"title": "Basel III and DORA"},
        ),
        score=1.0,
    )
]


def _model(**overrides: object) -> LearnedGateModel:
    values: dict[str, object] = {
        "feature_names": list(FEATURE_NAMES),
        "means": [0.0, 0.0, 0.0, 0.0],
        "scales": [1.0, 1.0, 1.0, 1.0],
        "weights": [2.0, 0.0, -3.0, 0.0],
        "bias": -1.0,
        "threshold": 0.5,
        "embedder": "hash-384",
    }
    values.update(overrides)
    return LearnedGateModel(**values)  # type: ignore[arg-type]


def test_specific_terms_are_names_and_numbers() -> None:
    """Capitalised words (not the first) and numbers are specific; stopwords are not."""
    assert specific_terms("What was Deutsche Bank's CET1 ratio in 2024?") == [
        "deutsche",
        "bank",
        "cet1",
        "2024",
    ]
    assert specific_terms("How does the leverage ratio work?") == []


def test_compute_features_counts_titles_as_evidence() -> None:
    """Coverage, best sentence, missing names (title counts) and similarity."""
    feats = compute_features("What is the leverage ratio minimum under DORA?", CTX)
    assert feats["coverage_top3"] == pytest.approx(1.0)
    assert feats["missing_specific_terms"] == 0.0  # 'DORA' appears in the title
    assert feats["semantic_similarity"] == 0.0  # no embedder given
    feats = compute_features("What is Commerzbank's leverage ratio?", CTX, HashEmbedder(64))
    assert feats["missing_specific_terms"] == 1.0
    assert 0.0 < feats["semantic_similarity"] <= 1.0
    assert compute_features("anything", [])["coverage_top3"] == 0.0


def test_rule_gate() -> None:
    """The rule gate answers iff coverage reaches the threshold."""
    gate = RuleGate(0.5)
    assert gate.decide({"coverage_top3": 0.5}).answer
    decision = gate.decide({"coverage_top3": 0.2})
    assert not decision.answer and "20%" in decision.reason


def test_learned_gate_probability_and_round_trip(tmp_path: Path) -> None:
    """P = sigmoid(bias + w·z); the model survives a JSON round trip."""
    gate = LearnedGate(_model())
    feats = {
        "coverage_top3": 1.0,
        "best_sentence_coverage": 0.0,
        "missing_specific_terms": 0.0,
        "semantic_similarity": 0.0,
    }
    assert gate.probability(feats) == pytest.approx(1 / (1 + np.exp(-1.0)))
    assert gate.decide(feats).answer
    feats["missing_specific_terms"] = 1.0
    decision = gate.decide(feats)
    assert not decision.answer and "learned gate" in decision.reason

    gate.save(tmp_path / "g.json")
    assert LearnedGate.load(tmp_path / "g.json").model == gate.model
    assert gate.compatible_with("hash-384 (fallback)")
    assert not gate.compatible_with("all-MiniLM-L6-v2")
    with pytest.raises(ValueError, match="features"):
        LearnedGate(_model(feature_names=["a", "b", "c", "d"]))


def test_cost_and_threshold_hand_computed() -> None:
    """Expected cost weights false answers 3x; the threshold minimises it."""
    labels = np.array([1, 1, 0, 0])
    answer = np.array([True, False, True, False])
    assert expected_cost(labels, answer, 3.0, 1.0) == pytest.approx((3 + 1) / 4)
    probs = np.array([0.9, 0.8, 0.3, 0.1])
    threshold = choose_threshold(probs, labels, 3.0, 1.0)
    assert 0.3 < threshold <= 0.8
    assert expected_cost(labels, probs >= threshold, 3.0, 1.0) == 0.0


def _sample(answerable: bool, coverage: float, kind: str = "") -> Sample:
    example = QAExample(question="q?", relevant_doc_ids=["d"] if answerable else [], kind=kind)
    return Sample(
        example,
        {
            "coverage_top3": coverage,
            "best_sentence_coverage": coverage,
            "missing_specific_terms": 0.0 if answerable else 1.0,
            "semantic_similarity": coverage,
        },
    )


def test_train_separates_classes_and_metrics() -> None:
    """On separable data the trained gate makes no mistakes."""
    rng = np.random.default_rng(0)
    samples = [_sample(True, float(rng.uniform(0.6, 1.0))) for _ in range(20)]
    samples += [_sample(False, float(rng.uniform(0.0, 0.4))) for _ in range(10)]
    gate, cv_prob = train(samples, 3.0, 1.0, embedder_name="hash-384")
    assert len(cv_prob) == 30
    assert gate.model.trained_on == {"answerable": 20, "unanswerable": 10}
    assert gate.model.embedder == "hash-384"
    decisions = [gate.decide(s.features).answer for s in samples]
    metrics = gate_metrics(samples, decisions, 3.0, 1.0)
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["false_refusal_rate"] == 0.0
    assert metrics["expected_cost"] == 0.0


def test_train_gate_cli_writes_model_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The training script runs end to end on the shipped dev/test sets."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    out = tmp_path / "gate.json"
    assert train_main(["--out", str(out), "--report", str(tmp_path), "--hash-embedder"]) == 0
    gate = LearnedGate.load(out)
    assert gate.model.trained_on["unanswerable"] >= 16
    report = (tmp_path / "gate_report.md").read_text(encoding="utf-8")
    assert "Learned gate, held-out test" in report


def test_load_gate_modes(tmp_path: Path) -> None:
    """rule/auto/learned selection, missing files and embedder mismatches."""
    path = tmp_path / "gate.json"
    LearnedGate(_model()).save(path)

    def settings(mode: str, model_path: Path = path) -> Settings:
        return Settings(_env_file=None, refusal_gate=mode, gate_model_path=str(model_path))

    assert isinstance(load_gate(settings("rule"), "hash-384"), RuleGate)
    assert isinstance(load_gate(settings("auto"), "hash-384"), LearnedGate)
    assert isinstance(load_gate(settings("auto"), "all-MiniLM-L6-v2"), RuleGate)  # mismatch
    assert isinstance(load_gate(settings("auto", tmp_path / "none.json"), "x"), RuleGate)
    with pytest.raises(FileNotFoundError):
        load_gate(settings("learned", tmp_path / "none.json"), "x")
    with pytest.raises(ValueError, match="retrain"):
        load_gate(settings("learned"), "all-MiniLM-L6-v2")
    with pytest.raises(ValueError, match="REFUSAL_GATE"):
        load_gate(settings("sometimes"), "x")
