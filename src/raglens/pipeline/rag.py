"""Retrieve -> (rerank) -> prompt -> generate, with inline citations and refusal gating."""

import re
import time
from typing import Any

from pydantic import BaseModel, Field

from raglens.embed.base import BaseEmbedder
from raglens.llm.base import REFUSAL_TEXT, BaseLLM, GenerationRequest
from raglens.pipeline.gate import RefusalGate, RuleGate, compute_features
from raglens.rerank.cross_encoder import BaseReranker
from raglens.retrieve.base import BaseRetriever, SearchResult
from raglens.textutil import content_stems, sentence_spans, split_sentences

SYSTEM_PROMPT = (
    "You are a precise assistant that answers strictly from the numbered context passages. "
    "Cite every claim with the passage number in square brackets, e.g. [1] or [2]. "
    "Do not use outside knowledge. If the context does not contain the answer, reply exactly: "
    f'"{REFUSAL_TEXT}"'
)

_CITATION_RE = re.compile(r"\[(\d+)\]")
# Bracketed citation groups real LLMs emit: [1, 2], [1-3], [1 and 2], [Source 1], [passage 2; 3].
_CITATION_GROUP_RE = re.compile(
    r"\[\s*(?:(?:source|passage|doc(?:ument)?|context)\s*)?"
    r"(\d+(?:\s*(?:,|;|-|\u2013|and|&)\s*(?:(?:source|passage|doc(?:ument)?|context)\s*)?\d+)*)\s*\]",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"\b(?:i\s+(?:do\s+not|don't|cannot|can't)\s+(?:know|answer|find)"
    r"|(?:cannot|can't|unable\s+to)\s+(?:answer|find|determine)"
    r"|no\s+(?:relevant\s+)?information"
    r"|(?:do|does)\s+not\s+(?:contain|mention|provide|include|cover|specify|say)"
    r"|(?:doesn't|don't)\s+(?:contain|mention|provide|include|cover|specify|say)"
    r"|not\s+(?:mentioned|covered|addressed|specified)\s+in)",
    re.IGNORECASE,
)
SUPPORT_THRESHOLD = 0.5
MAX_CITATION_RANGE = 20
MAX_CITATION_NUMBER = 50


class Citation(BaseModel):
    """One citation marker resolved to its source chunk and the sentence that supports it."""

    index: int = Field(..., description="Citation number as it appears in the answer")
    chunk_id: str
    doc_id: str
    title: str = ""
    source_path: str = ""
    page_number: int = 1
    evidence: str = Field(
        "", description="Sentence in the cited chunk that best supports the claim"
    )
    evidence_start: int = Field(-1, description="Start offset of the evidence in the chunk text")
    evidence_end: int = Field(-1, description="End offset of the evidence in the chunk text")
    support: float = Field(
        0.0, description="Share of the claim's content words found in the evidence (0-1)"
    )

    @property
    def supported(self) -> bool:
        """True when at least half of the claim's content words appear in the evidence."""
        return self.support >= SUPPORT_THRESHOLD


class RAGAnswer(BaseModel):
    """Result of one question through the pipeline."""

    question: str
    answer: str
    refused: bool = False
    refusal_reason: str = ""
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[SearchResult] = Field(default_factory=list)
    retriever: str = ""
    reranked: bool = False
    llm: str = ""
    top_score: float = 0.0
    query_coverage: float = 0.0
    gate: str = Field("", description="Refusal gate used (RuleGate or LearnedGate)")
    gate_score: float | None = Field(
        None, description="Gate confidence that the question is answerable (0-1)"
    )
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list, description="Quality warnings for the answer")


def build_prompt(question: str, contexts: list[SearchResult]) -> str:
    """Format the question and numbered context passages into the user prompt."""
    blocks = []
    for i, result in enumerate(contexts, start=1):
        title = result.chunk.metadata.get("title", result.chunk.doc_id)
        blocks.append(f"[{i}] (source: {title})\n{result.chunk.text.strip()}")
    context_text = "\n\n".join(blocks)
    return (
        f"Context passages:\n\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above and cite passages like [1]."
    )


def claims_by_citation(answer: str) -> dict[int, list[str]]:
    """Map each citation number to the answer sentences (markers removed) that cite it."""
    claims: dict[int, list[str]] = {}
    for sentence in split_sentences(answer):
        claim = _CITATION_RE.sub("", sentence).strip()
        for match in _CITATION_RE.finditer(sentence):
            claims.setdefault(int(match.group(1)), []).append(claim)
    return claims


def find_evidence(claims: list[str], chunk_text: str) -> tuple[int, int, float]:
    """Locate the chunk sentence that best supports any of ``claims``.

    Support is the share of a claim's distinct content stems that occur in the sentence. Returns
    ``(start, end, support)`` offsets into ``chunk_text``, or ``(-1, -1, 0.0)`` if nothing
    overlaps.
    """
    best = (-1, -1, 0.0)
    spans = sentence_spans(chunk_text)
    span_stems = [set(content_stems(chunk_text[s:e])) for s, e in spans]
    for claim in claims:
        claim_stems = set(content_stems(claim))
        if not claim_stems:
            continue
        for (start, end), stems in zip(spans, span_stems, strict=True):
            score = len(claim_stems & stems) / len(claim_stems)
            if score > best[2]:
                best = (start, end, score)
    return best


def extract_citations(answer: str, contexts: list[SearchResult]) -> list[Citation]:
    """Resolve ``[n]`` markers to chunks and attach the supporting evidence sentence.

    Out-of-range and repeated markers are dropped; order of first appearance is kept.
    """
    claims = claims_by_citation(answer)
    seen: set[int] = set()
    citations: list[Citation] = []
    for match in _CITATION_RE.finditer(answer):
        idx = int(match.group(1))
        if idx in seen or not 1 <= idx <= len(contexts):
            continue
        seen.add(idx)
        chunk = contexts[idx - 1].chunk
        start, end, support = find_evidence(claims.get(idx, []), chunk.text)
        citations.append(
            Citation(
                index=idx,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=str(chunk.metadata.get("title", "")),
                source_path=str(chunk.metadata.get("source_path", "")),
                page_number=chunk.page_number,
                evidence=chunk.text[start:end] if start >= 0 else "",
                evidence_start=start,
                evidence_end=end,
                support=round(support, 4),
            )
        )
    return citations


def normalize_citations(answer: str) -> str:
    """Rewrite citation groups like ``[1, 2]``, ``[1-3]`` or ``[Source 2]`` as ``[1][2]...``.

    Groups with numbers above ``MAX_CITATION_NUMBER`` or ranges longer than
    ``MAX_CITATION_RANGE`` are left untouched; they are years or other figures, not citations.
    """

    def expand(match: re.Match[str]) -> str:
        numbers: list[int] = []
        for part in re.split(r"\s*(?:,|;|and|&)\s*", match.group(1), flags=re.IGNORECASE):
            part = re.sub(r"[^\d\-\u2013]", "", part)
            bounds = [int(n) for n in re.split(r"[-\u2013]", part) if n]
            if len(bounds) == 2 and 0 < bounds[1] - bounds[0] <= MAX_CITATION_RANGE:
                numbers.extend(range(bounds[0], bounds[1] + 1))
            elif len(bounds) == 1:
                numbers.append(bounds[0])
            else:
                return match.group(0)
        if any(n < 1 or n > MAX_CITATION_NUMBER for n in numbers):
            return match.group(0)  # e.g. a year range like [2019-2025], not a citation
        return "".join(f"[{n}]" for n in dict.fromkeys(numbers))

    return _CITATION_GROUP_RE.sub(expand, answer)


def is_refusal(answer: str) -> bool:
    """True for the standard refusal text, or an uncited answer phrased as a refusal.

    Real LLMs rarely copy the refusal string exactly ("The documents do not mention..."), so any
    answer without citation markers that matches a refusal phrase counts. An answer that cites
    passages is treated as an answer even if it says part of the question is not covered.
    """
    text = answer.strip()
    if text.lower().startswith(REFUSAL_TEXT.lower().rstrip(".")):
        return True
    return not _CITATION_RE.search(text) and bool(_REFUSAL_RE.search(text))


class RAGPipeline:
    """End-to-end question answering over a retriever.

    The relevance gate refuses (without calling the LLM) when either the best first-stage
    retrieval score is below ``min_score`` or the top ``context_k`` passages cover less than
    ``min_query_coverage`` of the question's content words. The second check is retriever
    agnostic, which matters because BM25, cosine and RRF scores live on different scales.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        reranker: BaseReranker | None = None,
        *,
        retriever_name: str = "",
        top_k: int = 5,
        rerank_candidates: int = 20,
        min_score: float = 0.01,
        min_query_coverage: float = 0.5,
        gate: RefusalGate | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.retriever_name = retriever_name or type(retriever).__name__
        self.top_k = top_k
        self.rerank_candidates = max(rerank_candidates, top_k)
        self.min_score = min_score
        self.min_query_coverage = min_query_coverage
        self.gate = gate or RuleGate(min_query_coverage)
        self.embedder = embedder

    def retrieve(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[SearchResult], dict[str, float]]:
        """Run first-stage retrieval and optional reranking; returns results and timings."""
        k = top_k or self.top_k
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        pool = self.retriever.retrieve(
            question, top_k=self.rerank_candidates if self.reranker else k
        )
        timings["retrieve"] = (time.perf_counter() - t0) * 1000.0

        if self.reranker is not None:
            t0 = time.perf_counter()
            pool = self.reranker.rerank(question, pool, top_k=k)
            timings["rerank"] = (time.perf_counter() - t0) * 1000.0
        return pool[:k], timings

    def ask(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """Answer ``question`` with citations, or refuse when evidence is insufficient."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        contexts, timings = self.retrieve(question, top_k)
        top_score = max(
            (float(r.metadata.get("pre_rerank_score", r.score)) for r in contexts), default=0.0
        )
        features = compute_features(
            question, contexts, self.embedder if self.gate.needs_embedder else None
        )
        coverage = features["coverage_top3"]
        base: dict[str, Any] = {
            "question": question,
            "contexts": contexts,
            "retriever": self.retriever_name,
            "reranked": self.reranker is not None,
            "llm": self.llm.name,
            "top_score": top_score,
            "query_coverage": coverage,
            "gate": type(self.gate).__name__,
        }

        reason = ""
        if not contexts:
            reason = "no passages retrieved"
        elif top_score < self.min_score:
            reason = f"best retrieval score {top_score:.4f} < threshold {self.min_score}"
        else:
            decision = self.gate.decide(features)
            base["gate_score"] = decision.score
            if not decision.answer:
                reason = decision.reason
        if reason:
            timings["generate"] = 0.0
            return RAGAnswer(
                answer=REFUSAL_TEXT, refused=True, refusal_reason=reason, timings_ms=timings, **base
            )

        request = GenerationRequest(
            system=SYSTEM_PROMPT,
            prompt=build_prompt(question, contexts),
            question=question,
            contexts=[r.chunk.text for r in contexts],
        )
        t0 = time.perf_counter()
        answer = normalize_citations(self.llm.generate(request))
        timings["generate"] = (time.perf_counter() - t0) * 1000.0

        refused = is_refusal(answer)
        citations = [] if refused else extract_citations(answer, contexts)
        warnings: list[str] = []
        if not refused and not citations:
            warnings.append("answer has no valid citations")
        elif any(not c.supported for c in citations):
            warnings.append("some citations are only weakly supported by their passage")
        return RAGAnswer(
            answer=answer,
            refused=refused,
            refusal_reason="model declined: context did not contain the answer" if refused else "",
            citations=citations,
            warnings=warnings,
            timings_ms=timings,
            **base,
        )
