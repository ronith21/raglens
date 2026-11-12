"""Retrieval and answer-quality metrics. Every function is small enough to check by hand."""

import math
import random
from collections.abc import Sequence

from raglens.retrieve.base import SearchResult
from raglens.textutil import content_tokens, extract_citations, stemmed_content


def doc_ranking(results: Sequence[SearchResult]) -> list[str]:
    """Collapse a chunk ranking into a document ranking (first occurrence wins).

    Questions are labelled with relevant *documents*, so two chunks from the same file must not
    count twice, and a chunk ranked 1st and its sibling ranked 2nd both mean "document at rank 1".
    """
    ranking: list[str] = []
    for r in results:
        source = str(r.chunk.metadata.get("source", r.chunk.doc_id))
        if source not in ranking:
            ranking.append(source)
    return ranking


def hit_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant document is in the top k, else 0.0."""
    return 1.0 if relevant & set(ranking[:k]) else 0.0


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of relevant documents found in the top k."""
    if not relevant:
        return 0.0
    return len(relevant & set(ranking[:k])) / len(relevant)


def reciprocal_rank(ranking: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """1 / rank of the first relevant document within the top k, or 0."""
    for i, doc in enumerate(ranking[:k], start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k: DCG uses gain 1 / log2(rank + 1), normalised by the ideal DCG."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranking[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def citation_precision(
    answer: str, cited_sources: Sequence[str], relevant: set[str]
) -> float | None:
    """Share of cited passages that come from a relevant document (None if nothing is cited)."""
    if not extract_citations(answer) or not cited_sources:
        return None
    return sum(1 for s in cited_sources if s in relevant) / len(cited_sources)


def groundedness(answer: str, cited_texts: Sequence[str]) -> float | None:
    """Share of the answer's content words that appear in the passages it cites.

    1.0 means every substantive word is supported by the cited evidence; a model that adds
    outside facts (extrinsic hallucination) scores lower. None when there is nothing cited.
    """
    answer_terms = stemmed_content(_strip_markers(answer))
    if not answer_terms or not cited_texts:
        return None
    support: set[str] = set()
    for text in cited_texts:
        support |= stemmed_content(text)
    return len(answer_terms & support) / len(answer_terms)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between the answer and the hand-written reference answer."""
    pred = content_tokens(_strip_markers(prediction))
    ref = content_tokens(reference)
    if not pred or not ref:
        return 0.0
    common: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t in pred:
        if ref_counts.get(t, 0) > common.get(t, 0):
            common[t] = common.get(t, 0) + 1
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..100); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of (a - b) and its 95% bootstrap confidence interval, resampling questions.

    If the interval excludes 0, the difference between two systems on this question set is
    unlikely to be luck of which questions were asked.
    """
    if len(a) != len(b) or not a:
        raise ValueError("a and b must be non-empty and the same length")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return sum(diffs) / n, means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]


def _strip_markers(text: str) -> str:
    for n in extract_citations(text):
        text = text.replace(f"[{n}]", " ")
    return text
