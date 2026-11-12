"""End-to-end pipeline tests: citations, refusals, modes, reranking, building from disk."""

from pathlib import Path

import pytest

from raglens.config import Settings
from raglens.embed.lsa import LSAEmbedder
from raglens.llm.base import BaseLLM, GenerationRequest
from raglens.pipeline.builder import build_pipeline, load_chunks
from raglens.pipeline.rag import RAGPipeline, evidence_score
from raglens.rerank.cross_encoder import BaseReranker
from raglens.retrieve.base import SearchResult
from raglens.retrieve.store import IndexStore
from raglens.textutil import REFUSAL, extract_citations, is_refusal, split_sentences, stem

from .conftest import CORPUS


def test_answer_is_cited_and_grounded(pipeline: RAGPipeline) -> None:
    """A corpus question gets an answer whose citations point at retrieved passages."""
    result = pipeline.ask("What is the Reciprocal Rank Fusion formula?")
    assert not result.refused
    assert result.citations
    assert result.contexts[0].source == "12_hybrid_retrieval_rrf"
    for cit in result.citations:
        assert result.contexts[cit.number - 1].chunk_id == cit.chunk_id
    assert set(result.timings_ms) >= {"retrieve", "generate", "total"}


def test_unanswerable_question_is_refused(pipeline: RAGPipeline) -> None:
    """Off-topic questions are refused before calling the generator."""
    result = pipeline.ask("Who won the 2018 FIFA World Cup?")
    assert result.refused
    assert result.answer == REFUSAL
    assert result.evidence_score < pipeline.refusal_threshold
    assert result.timings_ms["generate"] == 0.0


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_every_retriever_mode_works(pipeline: RAGPipeline, mode: str) -> None:
    """All three retrievers return labelled results."""
    result = pipeline.ask("How does BM25 saturate term frequency?", mode=mode, top_k=3)  # type: ignore[arg-type]
    assert len(result.contexts) == 3
    assert result.retriever == mode


def test_hybrid_contexts_explain_both_rankings(pipeline: RAGPipeline) -> None:
    """Hybrid results carry each retriever's rank so the UI can show why a passage won."""
    ctx = pipeline.ask("inverted index postings list").contexts[0]
    assert "rrf_score" in ctx.signals
    assert {"bm25_rank", "dense_rank"} & set(ctx.signals)


def test_invalid_inputs(pipeline: RAGPipeline) -> None:
    """Empty questions, unknown modes and bad thresholds raise ValueError."""
    with pytest.raises(ValueError):
        pipeline.ask("   ")
    with pytest.raises(ValueError):
        pipeline.retrieve("q", mode="sparse")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RAGPipeline(pipeline.retriever, pipeline.llm, refusal_threshold=1.5)


class _CitesBadly(BaseLLM):
    name = "bad"

    def generate(self, request: GenerationRequest) -> str:
        return "Something [1] and something else [9]."


class _Abstains(BaseLLM):
    name = "abstain"

    def generate(self, request: GenerationRequest) -> str:
        return "I don't know based on the provided documents."


def test_out_of_range_citations_are_flagged(pipeline: RAGPipeline) -> None:
    """Citation numbers with no matching passage are reported, not silently accepted."""
    p = RAGPipeline(pipeline.retriever, _CitesBadly(), refusal_threshold=0.0)
    result = p.ask("gradient descent learning rate", top_k=3)
    assert [c.number for c in result.citations] == [1]
    assert result.invalid_citations == [9]


def test_model_abstention_is_a_refusal(pipeline: RAGPipeline) -> None:
    """If the model itself abstains, the result is marked refused."""
    result = RAGPipeline(pipeline.retriever, _Abstains(), refusal_threshold=0.0).ask(
        "gradient descent"
    )
    assert result.refused and result.refusal_reason == "model abstained"


class _ReverseReranker(BaseReranker):
    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        return list(reversed(results))[:top_k]


def test_rerank_stage_is_timed_and_applied(pipeline: RAGPipeline) -> None:
    """Reranking pulls a wider pool, reorders it and records its own latency."""
    p = RAGPipeline(pipeline.retriever, pipeline.llm, reranker=_ReverseReranker(), rerank_pool=10)
    plain = p.ask("BM25 inverse document frequency", top_k=3)
    reranked = p.ask("BM25 inverse document frequency", top_k=3, rerank=True)
    assert "rerank" in reranked.timings_ms
    assert reranked.retriever == "hybrid+rerank"
    assert [c.chunk_id for c in reranked.contexts] != [c.chunk_id for c in plain.contexts]


def test_evidence_score_is_word_coverage() -> None:
    """Evidence = best share of the question's stemmed content words found in one passage."""
    from raglens.ingest.models import Chunk

    results = [
        SearchResult(
            chunk=Chunk(chunk_id="a", doc_id="a", text="embeddings map text to vectors"), score=1
        ),
        SearchResult(chunk=Chunk(chunk_id="b", doc_id="b", text="nothing relevant"), score=0.5),
    ]
    assert evidence_score("What do embedding vectors map?", results) == pytest.approx(1.0)
    assert evidence_score("bananas", results) == 0.0
    assert evidence_score("the", results) == 0.0


def test_text_helpers() -> None:
    """Sentence splitting skips headings and math; citations parse in order; stemming is light."""
    text = "# Title\n$$x=1$$\n- First point here. Second point (B) here.\nThird."
    assert split_sentences(text) == ["First point here.", "Second point (B) here.", "Third."]
    assert extract_citations("a [2] b [1] c [2]") == [2, 1]
    assert is_refusal("I don't know based on the provided documents.")
    assert stem("embeddings") == "embedd" and stem("bm25") == "bm25"


def test_build_pipeline_indexes_then_loads_saved_index(tmp_path: Path) -> None:
    """Without an index the corpus is indexed; with a saved index it is loaded from disk."""
    settings = Settings(
        corpus_dir=str(CORPUS), index_dir=str(tmp_path / "idx"), embedder="lsa", llm_provider="fake"
    )
    fresh = build_pipeline(settings)
    IndexStore.save(fresh.retriever, settings.index_dir)
    loaded = build_pipeline(settings, embedder=LSAEmbedder())
    q = "How does HNSW search work?"
    assert [c.chunk_id for c in fresh.ask(q).contexts] == [
        c.chunk_id for c in loaded.ask(q).contexts
    ]


def test_load_chunks_rejects_empty_folder(tmp_path: Path) -> None:
    """A folder without supported files is an error, not an empty index."""
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path, 512, 64)
