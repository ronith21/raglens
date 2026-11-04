"""Retrieval, answer-quality and latency metrics.

Retrieval metrics are computed at *document* level: the ranked chunk list is collapsed to the
ordered list of distinct documents (first occurrence wins) before Hit/Recall/MRR/nDCG are taken.
Relevance is binary, so nDCG uses gain 1 for a relevant document.
"""

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from raglens.textutil import content_stems, content_tokens


def dedupe_preserving_order(items: Sequence[str]) -> list[str]:
    """Remove duplicates while keeping the first occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant id appears in the top k, else 0.0."""
    return 1.0 if any(doc in relevant for doc in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of relevant ids found in the top k (0.0 when there are no relevant ids)."""
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    """1 / rank of the first relevant id within the top k (0.0 if none)."""
    window = ranked if k is None else ranked[:k]
    for position, doc in enumerate(window, start=1):
        if doc in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, doc in enumerate(ranked[:k], start=1)
        if doc in relevant
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def citation_precision(cited_doc_ids: Sequence[str], relevant: set[str]) -> float | None:
    """Share of cited chunks whose document is relevant; None when nothing was cited."""
    if not cited_doc_ids:
        return None
    return sum(1 for doc in cited_doc_ids if doc in relevant) / len(cited_doc_ids)


def groundedness(answer: str, cited_texts: Sequence[str]) -> float | None:
    """Share of the answer's content tokens that appear in the cited context.

    Citation markers like ``[1]`` contain no letters and are ignored automatically. Returns
    None when the answer has no content tokens or nothing was cited.
    """
    answer_tokens = content_tokens(answer)
    if not answer_tokens or not cited_texts:
        return None
    context_tokens = set(content_tokens(" ".join(cited_texts)))
    return sum(1 for t in answer_tokens if t in context_tokens) / len(answer_tokens)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 over stemmed content words (citation markers ignored).

    Multiset overlap: a word counts as many times as it appears in both texts. Returns 0.0 when
    either side has no content words.
    """
    pred = Counter(content_stems(prediction))
    ref = Counter(content_stems(reference))
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def reference_recall(prediction: str, reference: str) -> float | None:
    """Share of the reference answer's distinct content stems found in the prediction."""
    ref = set(content_stems(reference))
    if not ref:
        return None
    return len(ref & set(content_stems(prediction))) / len(ref)


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0-100) with linear interpolation; 0.0 for an empty sequence."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    return float(sum(values) / len(values)) if values else 0.0
