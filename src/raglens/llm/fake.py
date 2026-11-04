"""Deterministic, extractive FakeLLM used in tests, offline demos and the eval harness."""

from raglens.llm.base import REFUSAL_TEXT, BaseLLM, GenerationRequest
from raglens.textutil import content_stems, split_sentences, strip_markdown


class FakeLLM(BaseLLM):
    """Builds an answer from the context sentences that best match the question.

    Each context is cleaned of markdown/LaTeX and split into sentences. A sentence's score is the
    sum of the weights of the (stemmed) question terms it contains, where a term's weight is
    ``1 / number of candidate sentences containing it``. Rare, specific question words ("large",
    "warmup") therefore count more than words every sentence shares ("gradient", "descent").
    Ties break by context order, then sentence order. The best ``max_sentences`` sentences are
    returned in reading order, each followed by its citation ``[i]``; with no overlap at all the
    standard refusal text is returned. Output depends only on the input, so eval numbers are
    reproducible.
    """

    name = "fake"

    def __init__(self, max_sentences: int = 2, max_contexts: int = 3) -> None:
        if max_sentences <= 0 or max_contexts <= 0:
            raise ValueError("max_sentences and max_contexts must be positive")
        self.max_sentences = max_sentences
        self.max_contexts = max_contexts

    def generate(self, request: GenerationRequest) -> str:
        """Return an extractive answer with inline citations."""
        q_stems = set(content_stems(request.question))
        if not q_stems or not request.contexts:
            return REFUSAL_TEXT

        sentences: list[tuple[int, int, str, set[str]]] = []
        for ctx_idx, context in enumerate(request.contexts[: self.max_contexts]):
            for sent_idx, sentence in enumerate(split_sentences(strip_markdown(context))):
                if len(sentence) >= 25 and not sentence.rstrip().endswith(":"):
                    sentences.append((ctx_idx, sent_idx, sentence, set(content_stems(sentence))))

        doc_freq = {t: sum(1 for *_, stems in sentences if t in stems) for t in q_stems}
        candidates: list[tuple[float, int, int, str]] = []
        for ctx_idx, sent_idx, sentence, stems in sentences:
            score = sum(1.0 / doc_freq[t] for t in q_stems & stems)
            if score > 0:
                candidates.append((score, ctx_idx, sent_idx, sentence))
        if not candidates:
            return REFUSAL_TEXT

        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        chosen = sorted(candidates[: self.max_sentences], key=lambda c: (c[1], c[2]))
        parts = []
        for _, ctx_idx, _, sentence in chosen:
            body = sentence.rstrip()
            punct = body[-1] if body[-1] in ".!?" else "."
            parts.append(f"{body.rstrip('.!?')} [{ctx_idx + 1}]{punct}")
        return " ".join(parts)
