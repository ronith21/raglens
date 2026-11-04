"""Deterministic synthetic QA generation from chunks.

For each sampled chunk the most informative sentence (most distinct content words) becomes the
reference answer, and a question is templated from it:

* ``"<Subject> is/are <...>"`` sentences become ``"What is/are <subject>?"``;
* otherwise the question asks about the sentence's three rarest content words (corpus document
  frequency), which keeps the question specific to the source chunk.

The output is a cheap, labelled (``synthetic: true``) supplement to the hand-written set, useful
for regression-testing retrieval on new corpora. It is not a substitute for human questions.
"""

import random
import re
from collections import Counter

from raglens.eval.dataset import QAExample
from raglens.ingest.models import Chunk
from raglens.textutil import content_tokens, split_sentences, strip_markdown

_DEFINITION_RE = re.compile(
    r"^(?P<subject>[A-Z][\w\-() ]{2,60}?)\s+(?P<verb>is|are)\s+(?:a|an|the)?\s*\w", re.UNICODE
)


def _best_sentence(chunk: Chunk) -> str | None:
    """The sentence with the most distinct content tokens (min 8 words), if any."""
    candidates = [s for s in split_sentences(strip_markdown(chunk.text)) if len(s.split()) >= 8]
    if not candidates:
        return None
    return max(candidates, key=lambda s: (len(set(content_tokens(s))), -len(s)))


def generate_synthetic_qa(
    chunks: list[Chunk], num_questions: int = 20, seed: int = 13
) -> list[QAExample]:
    """Generate up to ``num_questions`` QA pairs, at most one per chunk, reproducibly."""
    if num_questions <= 0 or not chunks:
        return []

    doc_freq: Counter[str] = Counter()
    for chunk in chunks:
        doc_freq.update(set(content_tokens(chunk.text)))

    order = list(range(len(chunks)))
    random.Random(seed).shuffle(order)

    examples: list[QAExample] = []
    for idx in order:
        if len(examples) >= num_questions:
            break
        chunk = chunks[idx]
        sentence = _best_sentence(chunk)
        if sentence is None:
            continue

        match = _DEFINITION_RE.match(sentence)
        if match:
            question = f"What {match.group('verb')} {match.group('subject').strip()}?"
        else:
            rare = sorted(set(content_tokens(sentence)), key=lambda t: (doc_freq[t], t))[:3]
            if len(rare) < 2:
                continue
            question = f"What do the documents say about {', '.join(rare)}?"

        slug = chunk.metadata.get("doc_slug")
        examples.append(
            QAExample(
                id=f"syn{len(examples) + 1:03d}",
                question=question,
                answer=sentence,
                relevant_doc_ids=[slug if isinstance(slug, str) and slug else chunk.doc_id],
                synthetic=True,
            )
        )
    return examples
