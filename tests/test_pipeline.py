"""Tests for the RAG pipeline: prompting, citations, refusal gating, reranking, timings."""

import pytest

from raglens.embed.hash import HashEmbedder
from raglens.ingest.models import Chunk
from raglens.llm.base import REFUSAL_TEXT, BaseLLM, GenerationRequest
from raglens.llm.fake import FakeLLM
from raglens.pipeline.rag import (
    RAGPipeline,
    build_prompt,
    claims_by_citation,
    extract_citations,
    find_evidence,
    is_refusal,
    normalize_citations,
)
from raglens.rerank.cross_encoder import BaseReranker
from raglens.retrieve.base import SearchResult
from raglens.retrieve.hybrid import HybridRetriever

CHUNKS = [
    Chunk(
        chunk_id="gd_0",
        doc_id="gd",
        text="Gradient descent updates parameters against the gradient. A learning rate that is "
        "too large makes updates overshoot minima and diverge.",
        metadata={
            "title": "Gradient Descent",
            "source_path": "corpus/01_gd.md",
            "doc_slug": "01_gd",
        },
    ),
    Chunk(
        chunk_id="rrf_0",
        doc_id="rrf",
        text="Reciprocal rank fusion combines BM25 and dense rankings using only rank positions, "
        "so no score calibration is required.",
        metadata={
            "title": "Hybrid Retrieval",
            "source_path": "corpus/12_rrf.md",
            "doc_slug": "12_rrf",
        },
    ),
    Chunk(
        chunk_id="tok_0",
        doc_id="tok",
        text="Byte-pair encoding merges the most frequent adjacent symbol pair repeatedly.",
        metadata={"title": "Tokenization", "source_path": "corpus/07_tok.md", "doc_slug": "07_tok"},
    ),
]


class RecordingLLM(BaseLLM):
    """Returns a canned answer and records the request it saw."""

    name = "recording"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> str:
        self.requests.append(request)
        return self.answer


class ReverseReranker(BaseReranker):
    """Reverses the candidate list (deterministic stand-in for a cross-encoder)."""

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        return [r.model_copy(update={"rank": i}) for i, r in enumerate(reversed(results), 1)][
            :top_k
        ]


@pytest.fixture
def retriever() -> HybridRetriever:
    return HybridRetriever(embedder=HashEmbedder(dimension=128), chunks=CHUNKS)


def test_answer_has_resolved_citations_and_timings(retriever: HybridRetriever) -> None:
    """A supported question gets an extractive answer whose [n] markers map to chunks."""
    pipe = RAGPipeline(retriever, FakeLLM(), retriever_name="hybrid", top_k=2)
    result = pipe.ask("What happens if the learning rate is too large?")

    assert not result.refused
    assert "overshoot" in result.answer
    assert result.citations[0].chunk_id == "gd_0"
    assert result.citations[0].title == "Gradient Descent"
    assert set(result.timings_ms) == {"retrieve", "generate"}
    assert result.retriever == "hybrid" and result.llm == "fake"
    assert result.query_coverage == 1.0


def test_refuses_off_topic_question_without_calling_llm(retriever: HybridRetriever) -> None:
    """Low query coverage short-circuits to the refusal string."""
    llm = RecordingLLM("should not be called")
    result = RAGPipeline(retriever, llm, min_query_coverage=0.5).ask("Who won the 2018 World Cup?")
    assert result.refused
    assert result.answer == REFUSAL_TEXT
    assert "cover" in result.refusal_reason
    assert llm.requests == []


def test_refuses_when_best_score_below_threshold(retriever: HybridRetriever) -> None:
    """The score gate refuses even when coverage would pass."""
    result = RAGPipeline(retriever, FakeLLM(), min_score=10.0).ask("reciprocal rank fusion")
    assert result.refused
    assert "threshold" in result.refusal_reason


def test_refuses_on_empty_index() -> None:
    """No passages means refusal, not an exception."""
    empty = HybridRetriever(embedder=HashEmbedder(dimension=16))
    result = RAGPipeline(empty, FakeLLM()).ask("anything at all")
    assert result.refused and result.refusal_reason == "no passages retrieved"


def test_model_refusal_is_detected(retriever: HybridRetriever) -> None:
    """If the LLM itself declines, the answer is flagged refused and has no citations."""
    llm = RecordingLLM(REFUSAL_TEXT + " [1]")
    result = RAGPipeline(retriever, llm).ask("How does reciprocal rank fusion work?")
    assert result.refused
    assert result.citations == []
    assert "model declined" in result.refusal_reason


def test_prompt_contains_numbered_context_and_question(retriever: HybridRetriever) -> None:
    """The LLM receives numbered passages, the question, and the citation instruction."""
    llm = RecordingLLM("Fusion uses ranks [1].")
    RAGPipeline(retriever, llm, top_k=2).ask("How does reciprocal rank fusion combine rankings?")
    request = llm.requests[0]
    assert request.prompt.startswith("Context passages:")
    assert "[1] (source: Hybrid Retrieval)" in request.prompt
    assert "Question: How does reciprocal rank fusion combine rankings?" in request.prompt
    assert "cite" in request.system.lower()
    assert len(request.contexts) == 2


def test_reranker_is_applied_and_timed(retriever: HybridRetriever) -> None:
    """The reranker reorders the candidate pool and adds a rerank timing."""
    pipe = RAGPipeline(retriever, FakeLLM(), ReverseReranker(), top_k=2, rerank_candidates=3)
    results, timings = pipe.retrieve("reciprocal rank fusion")
    assert "rerank" in timings
    assert results[0].chunk.chunk_id != "rrf_0"  # reversed: the best hit is no longer first
    assert pipe.ask("reciprocal rank fusion rankings").reranked


def test_extract_citations_ignores_out_of_range_and_duplicates() -> None:
    """[0], [9] and repeated markers are dropped; order of first appearance is kept."""
    contexts = [SearchResult(chunk=c, score=1.0, rank=i) for i, c in enumerate(CHUNKS, 1)]
    cites = extract_citations("A [2]. B [2][1]. C [0] [9].", contexts)
    assert [c.index for c in cites] == [2, 1]
    assert cites[0].chunk_id == "rrf_0"


def test_build_prompt_and_is_refusal_helpers() -> None:
    """Helper functions behave on edge inputs."""
    ctx = [SearchResult(chunk=CHUNKS[0], score=1.0)]
    assert "[1] (source: Gradient Descent)" in build_prompt("q?", ctx)
    assert is_refusal("  i don't know based on the provided documents")
    assert not is_refusal("I know this one [1].")


def test_validation_errors(retriever: HybridRetriever) -> None:
    """Blank questions and non-positive top_k are rejected."""
    with pytest.raises(ValueError, match="top_k"):
        RAGPipeline(retriever, FakeLLM(), top_k=0)
    with pytest.raises(ValueError, match="empty"):
        RAGPipeline(retriever, FakeLLM()).ask("   ")


def test_citations_carry_exact_evidence_span(retriever: HybridRetriever) -> None:
    """Each citation points at the supporting sentence by offsets into the chunk text."""
    result = RAGPipeline(retriever, FakeLLM(), top_k=2).ask(
        "What happens if the learning rate is too large?"
    )
    citation = result.citations[0]
    chunk_text = next(
        c.chunk.text for c in result.contexts if c.chunk.chunk_id == citation.chunk_id
    )
    assert citation.evidence == chunk_text[citation.evidence_start : citation.evidence_end]
    assert citation.evidence.startswith("A learning rate that is too large")
    assert citation.support == 1.0 and citation.supported


def test_unsupported_citation_is_flagged(retriever: HybridRetriever) -> None:
    """A claim the cited passage does not contain gets low support."""
    llm = RecordingLLM("Penguins migrate across Antarctica every winter [1].")
    result = RAGPipeline(retriever, llm, min_query_coverage=0.0).ask("reciprocal rank fusion")
    assert result.citations[0].support == 0.0
    assert not result.citations[0].supported
    assert result.citations[0].evidence == ""


def test_claims_by_citation_and_find_evidence() -> None:
    """Claims are grouped per marker; evidence picks the best-overlapping sentence."""
    claims = claims_by_citation("Fusion uses ranks [1][2]. Scores differ in scale [2].")
    assert claims == {
        1: ["Fusion uses ranks ."],
        2: ["Fusion uses ranks .", "Scores differ in scale ."],
    }
    text = "Intro sentence here. Fusion uses ranks only. Unrelated end."
    start, end, support = find_evidence(claims[1], text)
    assert text[start:end] == "Fusion uses ranks only."
    assert support == 1.0
    assert find_evidence(["the of"], text) == (-1, -1, 0.0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A [1, 2].", "A [1][2]."),
        ("B [1-3].", "B [1][2][3]."),
        ("C [Source 2] and [passage 1; 3].", "C [2] and [1][3]."),
        ("D [1 and 2].", "D [1][2]."),
        ("Years [2019-2025] stay.", "Years [2019-2025] stay."),
        ("Big [1-90] stays.", "Big [1-90] stays."),
    ],
)
def test_normalize_citations(raw: str, expected: str) -> None:
    """LLM citation styles become canonical [n] markers; years and huge ranges are untouched."""
    assert normalize_citations(raw) == expected


@pytest.mark.parametrize(
    ("answer", "refused"),
    [
        ("I don't know based on the provided documents.", True),
        ("The documents do not mention the capital of Australia.", True),
        ("I cannot answer that from the given context.", True),
        ("There is no information about this in the passages.", True),
        ("BM25 uses k1 [1]. The documents do not mention b.", False),
        ("Paris is the capital of France.", False),
    ],
)
def test_is_refusal_handles_paraphrased_refusals(answer: str, refused: bool) -> None:
    """Uncited refusal phrasing counts as a refusal; cited partial answers do not."""
    assert is_refusal(answer) is refused


def test_real_llm_style_output_is_normalized(retriever: HybridRetriever) -> None:
    """[1, 2] citations are parsed; an uncited answer is flagged with a warning."""
    cited = RAGPipeline(retriever, RecordingLLM("Fusion uses only ranks [1, 2]."), top_k=2).ask(
        "How does reciprocal rank fusion combine rankings?"
    )
    assert cited.answer == "Fusion uses only ranks [1][2]."
    assert [c.index for c in cited.citations] == [1, 2]

    uncited = RAGPipeline(retriever, RecordingLLM("Fusion uses ranks."), top_k=2).ask(
        "How does reciprocal rank fusion combine rankings?"
    )
    assert not uncited.refused
    assert uncited.warnings == ["answer has no valid citations"]

    paraphrased = RAGPipeline(
        retriever, RecordingLLM("The context does not mention that."), top_k=2
    ).ask("How does reciprocal rank fusion combine rankings?")
    assert paraphrased.refused and paraphrased.citations == []


def test_pipeline_reports_gate(retriever: HybridRetriever) -> None:
    """The answer records which gate ran and its score."""
    result = RAGPipeline(retriever, FakeLLM()).ask("How does reciprocal rank fusion work?")
    assert result.gate == "RuleGate"
    assert result.gate_score == pytest.approx(result.query_coverage)
