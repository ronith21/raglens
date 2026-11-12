"""End-to-end question answering: retrieve, (rerank), gate, generate, cite."""

import time
from typing import Literal

from pydantic import BaseModel, Field

from raglens.llm.base import BaseLLM, GenerationRequest
from raglens.rerank.cross_encoder import BaseReranker
from raglens.retrieve.base import SearchResult
from raglens.retrieve.hybrid import HybridRetriever
from raglens.textutil import REFUSAL, extract_citations, is_refusal, stemmed_content

RetrieverMode = Literal["bm25", "dense", "hybrid"]
RETRIEVER_MODES: tuple[RetrieverMode, ...] = ("bm25", "dense", "hybrid")


class RetrievedContext(BaseModel):
    """A passage shown to the model, with the signals that explain why it was retrieved."""

    number: int = Field(..., description="1-based passage number used in citations")
    chunk_id: str
    source: str = Field(..., description="Stable document key (file name without extension)")
    title: str
    page: int
    score: float
    text: str
    signals: dict[str, float] = Field(default_factory=dict)


class Citation(BaseModel):
    """A citation marker in the answer resolved to its passage."""

    number: int
    chunk_id: str
    source: str
    title: str


class RAGAnswer(BaseModel):
    """Full, inspectable result of one question."""

    question: str
    answer: str
    refused: bool
    refusal_reason: str | None = None
    evidence_score: float = Field(
        ..., description="Best share of question words found in a passage"
    )
    retriever: str
    llm: str
    citations: list[Citation] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    contexts: list[RetrievedContext] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


def evidence_score(question: str, results: list[SearchResult]) -> float:
    """Highest fraction of the question's (stemmed) content words contained in one passage.

    This is the refusal gate's signal. It is used instead of the raw retrieval score because
    RRF scores depend only on rank positions and say nothing about how well the top passage
    actually matches, and BM25 scores are unbounded and corpus dependent.
    """
    terms = stemmed_content(question)
    if not terms or not results:
        return 0.0
    return max(len(terms & stemmed_content(r.chunk.text)) / len(terms) for r in results)


class RAGPipeline:
    """Answers questions over an indexed corpus with grounded, cited answers."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: BaseLLM,
        reranker: BaseReranker | None = None,
        refusal_threshold: float = 0.5,
        rerank_pool: int = 20,
    ) -> None:
        if not 0.0 <= refusal_threshold <= 1.0:
            raise ValueError("refusal_threshold must be between 0 and 1")
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.refusal_threshold = refusal_threshold
        self.rerank_pool = rerank_pool

    def retrieve(
        self, question: str, mode: RetrieverMode = "hybrid", top_k: int = 5
    ) -> list[SearchResult]:
        """Run one of the three retrievers over the shared index."""
        if mode == "bm25":
            return self.retriever.bm25_retriever.retrieve(question, top_k=top_k)
        if mode == "dense":
            return self.retriever.dense_retriever.retrieve(question, top_k=top_k)
        if mode == "hybrid":
            return self.retriever.retrieve(question, top_k=top_k)
        raise ValueError(f"Unknown retriever mode {mode!r}; choose one of {RETRIEVER_MODES}")

    def ask(
        self,
        question: str,
        mode: RetrieverMode = "hybrid",
        top_k: int = 5,
        rerank: bool = False,
    ) -> RAGAnswer:
        """Answer ``question``; refuse when the retrieved evidence is too weak."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        pool = self.rerank_pool if (rerank and self.reranker) else top_k
        results = self.retrieve(question, mode=mode, top_k=max(pool, top_k))
        timings["retrieve"] = (time.perf_counter() - t0) * 1000

        if rerank and self.reranker is not None:
            t0 = time.perf_counter()
            results = self.reranker.rerank(question, results, top_k=top_k)
            timings["rerank"] = (time.perf_counter() - t0) * 1000
        results = results[:top_k]

        contexts = [_to_context(i, r) for i, r in enumerate(results, start=1)]
        score = evidence_score(question, results)
        label = f"{mode}+rerank" if rerank and self.reranker else mode

        if score < self.refusal_threshold:
            timings["generate"] = 0.0
            timings["total"] = (time.perf_counter() - t_start) * 1000
            return RAGAnswer(
                question=question,
                answer=REFUSAL,
                refused=True,
                refusal_reason=(
                    f"evidence score {score:.2f} is below the threshold "
                    f"{self.refusal_threshold:.2f}"
                ),
                evidence_score=score,
                retriever=label,
                llm=self.llm.name,
                contexts=contexts,
                timings_ms=timings,
            )

        t0 = time.perf_counter()
        answer = self.llm.generate(
            GenerationRequest(question=question, contexts=[c.text for c in contexts])
        )
        timings["generate"] = (time.perf_counter() - t0) * 1000

        citations: list[Citation] = []
        invalid: list[int] = []
        for n in extract_citations(answer):
            if 1 <= n <= len(contexts):
                ctx = contexts[n - 1]
                citations.append(
                    Citation(number=n, chunk_id=ctx.chunk_id, source=ctx.source, title=ctx.title)
                )
            else:
                invalid.append(n)

        refused = is_refusal(answer)
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return RAGAnswer(
            question=question,
            answer=answer,
            refused=refused,
            refusal_reason="model abstained" if refused else None,
            evidence_score=score,
            retriever=label,
            llm=self.llm.name,
            citations=citations,
            invalid_citations=invalid,
            contexts=contexts,
            timings_ms=timings,
        )


def _to_context(number: int, result: SearchResult) -> RetrievedContext:
    meta = result.chunk.metadata
    signals = {
        k: float(v)
        for k, v in result.metadata.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }
    return RetrievedContext(
        number=number,
        chunk_id=result.chunk.chunk_id,
        source=str(meta.get("source", result.chunk.doc_id)),
        title=str(meta.get("title", meta.get("source", result.chunk.doc_id))),
        page=result.chunk.page_number,
        score=result.score,
        text=result.chunk.text,
        signals=signals,
    )
