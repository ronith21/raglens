"""Train the learned refusal gate and compare it with the rule gate on held-out questions.

Usage::

    python -m raglens.eval.train_gate [--out data/models/refusal_gate.json] [--report docs/proof]

Training uses only the *dev* sets (the ML corpus questions and ``banking_dev.jsonl``). The decision
threshold is chosen on 5-fold cross-validated probabilities to minimise expected cost, where
answering an unanswerable question costs ``--cost-false-answer`` (default 3) and refusing an
answerable one costs ``--cost-false-refusal`` (default 1). The held-out ``banking_test.jsonl`` is
used once, for the final comparison. Requires scikit-learn (``pip install raglens[train]``).
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from raglens.chunking.splitter import RecursiveCharacterChunker
from raglens.config import Settings
from raglens.embed.base import BaseEmbedder
from raglens.engine import create_embedder
from raglens.eval.dataset import QAExample, load_dataset
from raglens.ingest.parser import ingest_directory
from raglens.pipeline.gate import (
    FEATURE_NAMES,
    LearnedGate,
    LearnedGateModel,
    RuleGate,
    compute_features,
)
from raglens.retrieve.hybrid import HybridRetriever

DEV_SETS = (
    ("data/sample_corpus", "data/eval/qa.jsonl"),
    ("data/corpora/banking", "data/eval/banking_dev.jsonl"),
)
TEST_SETS = (("data/corpora/banking", "data/eval/banking_test.jsonl"),)


@dataclass
class Sample:
    """Gate features and label for one question."""

    example: QAExample
    features: dict[str, float]

    @property
    def label(self) -> int:
        return 1 if self.example.answerable else 0


def collect(
    sets: tuple[tuple[str, str], ...], embedder: BaseEmbedder, chunk_size: int = 512
) -> list[Sample]:
    """Retrieve with the hybrid retriever and compute gate features for every question."""
    samples: list[Sample] = []
    for corpus, dataset in sets:
        chunks = RecursiveCharacterChunker(chunk_size, chunk_size // 8).chunk_documents(
            ingest_directory(corpus)
        )
        retriever = HybridRetriever(embedder=embedder, chunks=chunks)
        for example in load_dataset(dataset):
            contexts = retriever.retrieve(example.question, top_k=5)
            samples.append(Sample(example, compute_features(example.question, contexts, embedder)))
    return samples


def matrix(samples: list[Sample]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Feature matrix (in FEATURE_NAMES order) and labels."""
    x = np.array([[s.features[n] for n in FEATURE_NAMES] for s in samples], dtype=np.float64)
    y = np.array([s.label for s in samples], dtype=np.int64)
    return x, y


def expected_cost(
    labels: npt.NDArray[np.int64],
    answer: npt.NDArray[np.bool_],
    cost_false_answer: float,
    cost_false_refusal: float,
) -> float:
    """Average cost per question of a set of answer/refuse decisions."""
    false_answer = np.sum(answer & (labels == 0))
    false_refusal = np.sum(~answer & (labels == 1))
    return float(cost_false_answer * false_answer + cost_false_refusal * false_refusal) / len(
        labels
    )


def choose_threshold(
    probabilities: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    cost_false_answer: float,
    cost_false_refusal: float,
) -> float:
    """Threshold on P(answerable) with the lowest expected cost (ties: closest to 0.5)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    costs = [
        expected_cost(labels, probabilities >= t, cost_false_answer, cost_false_refusal)
        for t in grid
    ]
    best = min(costs)
    candidates = [t for t, c in zip(grid, costs, strict=True) if c == best]
    return float(min(candidates, key=lambda t: abs(t - 0.5)))


def train(
    samples: list[Sample],
    cost_false_answer: float,
    cost_false_refusal: float,
    seed: int = 0,
    embedder_name: str = "",
) -> tuple[LearnedGate, npt.NDArray[np.float64]]:
    """Fit the gate; returns it and the cross-validated dev probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x, y = matrix(samples)

    def make() -> object:
        return make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
        )

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_prob = cross_val_predict(make(), x, y, cv=folds, method="predict_proba")[:, 1]
    threshold = choose_threshold(cv_prob, y, cost_false_answer, cost_false_refusal)

    final = make()
    final.fit(x, y)  # type: ignore[attr-defined]
    scaler, logistic = final.steps[0][1], final.steps[1][1]  # type: ignore[attr-defined]
    model = LearnedGateModel(
        feature_names=list(FEATURE_NAMES),
        means=[float(v) for v in scaler.mean_],
        scales=[float(v) for v in scaler.scale_],
        weights=[float(v) for v in logistic.coef_[0]],
        bias=float(logistic.intercept_[0]),
        threshold=threshold,
        trained_on={"answerable": int(y.sum()), "unanswerable": int(len(y) - y.sum())},
        cost_false_answer=cost_false_answer,
        cost_false_refusal=cost_false_refusal,
        embedder=embedder_name,
    )
    return LearnedGate(model), np.asarray(cv_prob, dtype=np.float64)


def gate_metrics(
    samples: list[Sample], answer: list[bool], cost_fa: float, cost_fr: float
) -> dict[str, float]:
    """Refusal accuracy, false refusals (overall and on paraphrases) and expected cost."""
    labels = np.array([s.label for s in samples])
    ans = np.array(answer)
    unanswerable = labels == 0
    paraphrase = np.array([s.example.kind == "paraphrase" for s in samples])
    return {
        "refusal_accuracy": float(np.mean(~ans[unanswerable])) if unanswerable.any() else 0.0,
        "false_refusal_rate": float(np.mean(~ans[~unanswerable])) if (~unanswerable).any() else 0.0,
        "paraphrase_false_refusals": float(np.mean(~ans[paraphrase])) if paraphrase.any() else 0.0,
        "expected_cost": expected_cost(labels, ans, cost_fa, cost_fr),
    }


def render_report(
    gate: LearnedGate,
    rule: RuleGate,
    dev: list[Sample],
    cv_prob: npt.NDArray[np.float64],
    test: list[Sample],
    embedder_name: str,
    elapsed_s: float,
) -> str:
    """Markdown comparison of the rule gate and the learned gate."""
    m = gate.model
    fa, fr = m.cost_false_answer, m.cost_false_refusal
    rows = {
        "Rule gate (coverage >= 50%), dev": gate_metrics(
            dev, [rule.decide(s.features).answer for s in dev], fa, fr
        ),
        "Learned gate, dev (5-fold cross-validated)": gate_metrics(
            dev, [bool(p >= m.threshold) for p in cv_prob], fa, fr
        ),
        "**Rule gate, held-out test**": gate_metrics(
            test, [rule.decide(s.features).answer for s in test], fa, fr
        ),
        "**Learned gate, held-out test**": gate_metrics(
            test, [gate.decide(s.features).answer for s in test], fa, fr
        ),
    }
    lines = [
        "# Refusal gate report",
        "",
        f"Generated by `python -m raglens.eval.train_gate` in {elapsed_s:.1f}s with embedder "
        f"`{embedder_name}`. Every number below comes from that run.",
        "",
        f"- Training data (dev only): {m.trained_on['answerable']} answerable and "
        f"{m.trained_on['unanswerable']} unanswerable questions from "
        + ", ".join(f"`{d}`" for _, d in DEV_SETS),
        "- Held-out test: "
        + ", ".join(f"`{d}`" for _, d in TEST_SETS)
        + f" ({len(test)} questions,"
        " never used for training or threshold selection)",
        f"- Costs: answering an unanswerable question = {fa:g}, "
        f"refusing an answerable one = {fr:g}",
        f"- Chosen threshold on P(answerable): **{m.threshold:.2f}**",
        "",
        "| Gate | Refusal accuracy | False refusals | False refusals on paraphrases "
        "| Expected cost |",
        "|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        lines.append(
            f"| {name} | {r['refusal_accuracy']:.3f} | {r['false_refusal_rate']:.3f} | "
            f"{r['paraphrase_false_refusals']:.3f} | {r['expected_cost']:.3f} |"
        )
    lines += [
        "",
        "## Model",
        "",
        "Logistic regression on standardized features (`class_weight=balanced`). A positive "
        "weight means the feature makes the question look *more* answerable.",
        "",
        "| Feature | Weight | Mean | Std |",
        "|---|---|---|---|",
    ]
    for name, w, mean, scale in zip(m.feature_names, m.weights, m.means, m.scales, strict=True):
        lines.append(f"| `{name}` | {w:+.3f} | {mean:.3f} | {scale:.3f} |")
    lines += ["", f"Bias: {m.bias:+.3f}", ""]

    wrong = [s for s in test if gate.decide(s.features).answer != bool(s.label)]
    lines += ["## Held-out mistakes (learned gate)", ""]
    if not wrong:
        lines.append("None.")
    for s in wrong:
        decision = gate.decide(s.features)
        verdict = "answered" if decision.answer else "refused"
        lines.append(
            f"- `{s.example.id}` ({s.example.kind}, {verdict}, p={decision.score:.2f}): "
            f"{s.example.question}"
        )
    lines += [
        "",
        "> **Note:** one feature fix (counting document titles as evidence) was made after",
        "> inspecting a held-out mistake, so the held-out numbers are slightly optimistic. The",
        "> gate must be retrained whenever the embedder changes (`make gate`).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train and evaluate the learned refusal gate.")
    parser.add_argument("--out", default="data/models/refusal_gate.json")
    parser.add_argument("--report", default="docs/proof")
    parser.add_argument("--cost-false-answer", type=float, default=3.0)
    parser.add_argument("--cost-false-refusal", type=float, default=1.0)
    parser.add_argument("--hash-embedder", action="store_true")
    args = parser.parse_args(argv)

    overrides = {"use_hash_embedder": True} if args.hash_embedder else {}
    settings = Settings().model_copy(update=overrides)
    embedder = create_embedder(settings)

    start = time.perf_counter()
    dev = collect(DEV_SETS, embedder)
    test = collect(TEST_SETS, embedder)
    gate, cv_prob = train(
        dev, args.cost_false_answer, args.cost_false_refusal, embedder_name=embedder.name
    )
    gate.save(args.out)
    report = render_report(
        gate,
        RuleGate(settings.min_query_coverage),
        dev,
        cv_prob,
        test,
        embedder.name,
        time.perf_counter() - start,
    )
    report_path = Path(args.report) / "gate_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {args.out} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
