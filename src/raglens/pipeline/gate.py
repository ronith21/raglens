"""Refusal gates: decide whether retrieved evidence is good enough to answer.

Two gates share one interface:

* :class:`RuleGate` refuses when the top passages cover less than a fixed share of the
  question's content words (the original, hand-tuned rule).
* :class:`LearnedGate` is a logistic regression over four interpretable features, trained with
  ``python -m raglens.eval.train_gate``. Its weights, feature scaling and decision threshold are
  stored as plain JSON, so loading a model never executes code (unlike pickle) and does not need
  scikit-learn at inference time.
"""

import json
import math
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from raglens.embed.base import BaseEmbedder
from raglens.retrieve.base import SearchResult
from raglens.textutil import (
    STOPWORDS,
    content_stems,
    query_coverage,
    split_sentences,
    stem,
    strip_markdown,
    tokenize,
)

FEATURE_NAMES = (
    "coverage_top3",
    "best_sentence_coverage",
    "missing_specific_terms",
    "semantic_similarity",
)
_SPECIFIC_RE = re.compile(r"\b(?:[A-Z][\w\-]*|\w*\d[\w\-.]*)\b")


class GateDecision(BaseModel):
    """Outcome of a gate: answer or refuse, with a confidence and a readable reason."""

    answer: bool
    score: float = Field(..., description="Confidence that the question is answerable (0-1)")
    reason: str = ""
    features: dict[str, float] = Field(default_factory=dict)


def specific_terms(question: str) -> list[str]:
    """Names and numbers in the question (capitalised or containing a digit), excluding the
    first word and stopwords. These are the terms a grounded answer must actually find."""
    words = question.split()
    rest = " ".join(words[1:]) if words else ""
    terms = []
    for match in _SPECIFIC_RE.finditer(rest):
        token = match.group(0).strip(".-").lower()
        if token and token not in STOPWORDS and len(token) > 1:
            terms.append(token)
    return list(dict.fromkeys(terms))


def compute_features(
    question: str, contexts: list[SearchResult], embedder: BaseEmbedder | None = None
) -> dict[str, float]:
    """The four gate features for ``question`` and its retrieved ``contexts``.

    * coverage_top3: share of question content stems found in the top three passages (and
      their document titles);
    * best_sentence_coverage: the same share for the single best-matching sentence;
    * missing_specific_terms: share of names/numbers in the question absent from the passages;
    * semantic_similarity: highest cosine similarity between the question and a top-3 passage
      (0 when no embedder is given).
    """
    top = [r.chunk.text for r in contexts[:3]]
    # The prompt shows each passage with its document title, so titles count as evidence too.
    titles = [str(r.chunk.metadata.get("title", "")) for r in contexts[:3]]
    joined = " ".join(titles + top)
    q_stems = set(content_stems(question))

    best_sentence = 0.0
    if q_stems:
        for text in top:
            for sentence in split_sentences(strip_markdown(text)):
                overlap = len(q_stems & set(content_stems(sentence))) / len(q_stems)
                best_sentence = max(best_sentence, overlap)

    context_stems = {stem(t) for t in tokenize(joined)}
    context_tokens = set(tokenize(joined))
    terms = specific_terms(question)
    missing = [t for t in terms if t not in context_tokens and stem(t) not in context_stems]

    similarity = 0.0
    if embedder is not None and top:
        vectors = embedder.embed_texts([question, *top])
        similarity = float(np.max(vectors[1:] @ vectors[0]))

    return {
        "coverage_top3": query_coverage(question, joined),
        "best_sentence_coverage": best_sentence,
        "missing_specific_terms": len(missing) / len(terms) if terms else 0.0,
        "semantic_similarity": similarity,
    }


class RefusalGate(ABC):
    """Decides whether to answer from the retrieved contexts."""

    needs_embedder: bool = False

    @abstractmethod
    def decide(self, features: dict[str, float]) -> GateDecision:
        """Return the decision for precomputed features."""


class RuleGate(RefusalGate):
    """Refuse when top-3 coverage is below ``min_coverage``."""

    def __init__(self, min_coverage: float = 0.5) -> None:
        self.min_coverage = min_coverage

    def decide(self, features: dict[str, float]) -> GateDecision:
        """Answer iff coverage reaches the threshold."""
        coverage = features["coverage_top3"]
        ok = coverage >= self.min_coverage
        reason = (
            ""
            if ok
            else f"top passages cover {coverage:.0%} of the question's key terms "
            f"(< {self.min_coverage:.0%})"
        )
        return GateDecision(answer=ok, score=coverage, reason=reason, features=features)


class LearnedGateModel(BaseModel):
    """Serializable parameters of a logistic-regression gate."""

    feature_names: list[str]
    means: list[float]
    scales: list[float]
    weights: list[float]
    bias: float
    threshold: float
    trained_on: dict[str, int] = Field(default_factory=dict)
    embedder: str = Field("", description="Embedder used for training (similarity depends on it)")
    cost_false_answer: float = 3.0
    cost_false_refusal: float = 1.0


class LearnedGate(RefusalGate):
    """Logistic regression over standardized gate features with a cost-chosen threshold."""

    needs_embedder = True

    def __init__(self, model: LearnedGateModel) -> None:
        if list(model.feature_names) != list(FEATURE_NAMES):
            raise ValueError(f"Gate model features {model.feature_names} != {FEATURE_NAMES}")
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "LearnedGate":
        """Load a gate saved as JSON."""
        return cls(LearnedGateModel.model_validate_json(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        """Write the gate as pretty-printed JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.model.model_dump(), indent=2), encoding="utf-8")

    def probability(self, features: dict[str, float]) -> float:
        """P(answerable) from the logistic model."""
        m = self.model
        z = m.bias
        for name, mean, scale, weight in zip(
            m.feature_names, m.means, m.scales, m.weights, strict=True
        ):
            z += weight * (features[name] - mean) / (scale or 1.0)
        return 1.0 / (1.0 + math.exp(-z))

    def compatible_with(self, embedder_name: str) -> bool:
        """True if the gate was trained with the same embedder family (fallback suffix ignored)."""

        def base(name: str) -> str:
            return name.replace(" (fallback)", "").strip()

        return not self.model.embedder or base(self.model.embedder) == base(embedder_name)

    def decide(self, features: dict[str, float]) -> GateDecision:
        """Answer iff P(answerable) reaches the trained threshold."""
        p = self.probability(features)
        ok = p >= self.model.threshold
        reason = (
            ""
            if ok
            else f"learned gate: {p:.0%} confidence the documents answer this "
            f"(< {self.model.threshold:.0%} threshold)"
        )
        return GateDecision(answer=ok, score=p, reason=reason, features=features)
