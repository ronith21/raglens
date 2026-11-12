"""Synthetic QA generation: turn the most informative sentence of a chunk into a query.

This is a cheap sanity check ("can each retriever find the chunk a sentence came from?"),
not a substitute for the hand-written set: the query shares its exact words with the target,
so it favours lexical retrieval. The report labels it accordingly.
"""

import math
import random

from raglens.eval.dataset import QAItem
from raglens.ingest.models import Chunk
from raglens.textutil import content_tokens, split_sentences


def generate_synthetic(chunks: list[Chunk], n: int = 40, seed: int = 7) -> list[QAItem]:
    """Sample up to ``n`` chunks and use each one's highest-IDF sentence as the question."""
    doc_freq: dict[str, int] = {}
    for chunk in chunks:
        for term in set(content_tokens(chunk.text)):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    total = max(len(chunks), 1)

    def informativeness(sentence: str) -> float:
        terms = set(content_tokens(sentence))
        return sum(math.log(total / doc_freq.get(t, 1)) for t in terms)

    rng = random.Random(seed)
    sample = rng.sample(chunks, k=min(n, len(chunks)))
    items: list[QAItem] = []
    for chunk in sample:
        candidates = [s for s in split_sentences(chunk.text) if 6 <= len(content_tokens(s)) <= 40]
        if not candidates:
            continue
        sentence = max(candidates, key=informativeness)
        items.append(
            QAItem(
                id=f"syn{len(items) + 1:03d}",
                question=sentence,
                answer=sentence,
                relevant_doc_ids=[str(chunk.metadata.get("source", chunk.doc_id))],
                type="synthetic",
                split="test",
            )
        )
    return items
