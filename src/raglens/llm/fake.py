"""Deterministic extractive generator used in tests, CI and offline demos."""

from raglens.llm.base import BaseLLM, GenerationRequest
from raglens.textutil import REFUSAL, content_tokens, split_sentences


class FakeLLM(BaseLLM):
    """Builds the answer from the context sentences that best overlap with the question.

    It never invents text: every sentence it returns is copied from a passage and tagged with
    that passage's number, so it is a faithful lower bound for what a real model could cite.
    """

    name = "fake"

    def __init__(self, max_sentences: int = 2, min_overlap: int = 1) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be at least 1")
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, request: GenerationRequest) -> str:
        """Pick the highest-overlap sentences; refuse if none share a content word."""
        query_terms = set(content_tokens(request.question))
        if not query_terms or not request.contexts:
            return REFUSAL

        scored: list[tuple[float, int, int, str]] = []
        seen: set[str] = set()
        for ctx_idx, context in enumerate(request.contexts, start=1):
            for sent_idx, sentence in enumerate(split_sentences(context)):
                tokens = set(content_tokens(sentence))
                overlap = len(query_terms & tokens)
                if overlap < self.min_overlap or len(tokens) < 4 or sentence in seen:
                    continue
                seen.add(sentence)
                # Earlier passages win ties: they were ranked higher by retrieval.
                score = overlap / len(query_terms) - 0.01 * ctx_idx
                scored.append((score, ctx_idx, sent_idx, sentence))

        if not scored:
            return REFUSAL

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        # Extra sentences must be at least half as relevant as the best one, so an answer is
        # not padded with loosely related text (which would also hurt citation precision).
        best = scored[0][0]
        chosen = [item for item in scored[: self.max_sentences] if item[0] >= best / 2]
        chosen.sort(key=lambda item: (item[1], item[2]))
        return " ".join(f"{sentence} [{ctx_idx}]" for _, ctx_idx, _, sentence in chosen)
