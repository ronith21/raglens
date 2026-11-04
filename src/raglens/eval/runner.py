"""Run a pipeline over an evaluation dataset and aggregate metrics."""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from raglens.eval.dataset import QAExample
from raglens.eval.metrics import (
    citation_precision,
    dedupe_preserving_order,
    groundedness,
    hit_at_k,
    mean,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    reference_recall,
    token_f1,
)
from raglens.pipeline.rag import RAGPipeline

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


class QuestionResult(BaseModel):
    """Per-question outcome, kept for drill-down in reports and the UI."""

    id: str
    question: str
    kind: str = "direct"
    answerable: bool
    answer: str
    refused: bool
    retrieved_doc_slugs: list[str]
    relevant_doc_ids: list[str]
    reciprocal_rank: float
    citation_precision: float | None = None
    groundedness: float | None = None
    answer_f1: float | None = None
    answer_recall: float | None = None
    citations_supported: int = 0
    citations_total: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)


class EvalReport(BaseModel):
    """Aggregate metrics for one pipeline configuration."""

    config: str
    num_questions: int
    num_answerable: int
    num_unanswerable: int
    retrieval: dict[str, float] = Field(default_factory=dict)
    answer: dict[str, float] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    by_kind: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Per question type: n, hit@5, mrr, refused share"
    )
    questions: list[QuestionResult] = Field(default_factory=list)


def _doc_slug(metadata: dict[str, object], fallback: str) -> str:
    slug = metadata.get("doc_slug")
    return slug if isinstance(slug, str) and slug else fallback


def evaluate(
    pipeline: RAGPipeline,
    dataset: Sequence[QAExample],
    ks: Sequence[int] = DEFAULT_KS,
    config_name: str = "",
) -> EvalReport:
    """Evaluate retrieval (at max(ks) depth), answers, refusals and per-stage latency.

    Retrieval metrics use answerable questions only. Refusal accuracy is the share of
    unanswerable questions that were refused; the false-refusal rate is the share of answerable
    questions that were refused. Answer F1/recall compare each answerable question's answer with
    its reference answer (a wrong refusal scores 0). Citation support is the share of all
    citations whose evidence sentence covers at least half of the cited claim's content words.
    """
    depth = max(ks)
    results: list[QuestionResult] = []
    retrieval_scores: dict[str, list[float]] = {}
    stage_times: dict[str, list[float]] = {}
    cit_prec: list[float] = []
    grounded: list[float] = []
    answer_f1s: list[float] = []
    answer_recalls: list[float] = []
    support_hits = 0
    support_total = 0

    for example in dataset:
        relevant = set(example.relevant_doc_ids)

        ranked_results, _ = pipeline.retrieve(example.question, top_k=depth)
        ranked_docs = dedupe_preserving_order(
            [_doc_slug(r.chunk.metadata, r.chunk.doc_id) for r in ranked_results]
        )
        rr = reciprocal_rank(ranked_docs, relevant, depth)

        if example.answerable:
            for k in ks:
                retrieval_scores.setdefault(f"hit@{k}", []).append(
                    hit_at_k(ranked_docs, relevant, k)
                )
                retrieval_scores.setdefault(f"recall@{k}", []).append(
                    recall_at_k(ranked_docs, relevant, k)
                )
                retrieval_scores.setdefault(f"ndcg@{k}", []).append(
                    ndcg_at_k(ranked_docs, relevant, k)
                )
            retrieval_scores.setdefault("mrr", []).append(rr)

        rag = pipeline.ask(example.question)
        for stage, ms in rag.timings_ms.items():
            stage_times.setdefault(stage, []).append(ms)

        cited = {c.index: c for c in rag.citations}
        cited_slugs = [
            _doc_slug(rag.contexts[i - 1].chunk.metadata, cited[i].doc_id) for i in sorted(cited)
        ]
        cited_texts = [rag.contexts[i - 1].chunk.text for i in sorted(cited)]
        cp = citation_precision(cited_slugs, relevant) if example.answerable else None
        gr = groundedness(rag.answer, cited_texts) if not rag.refused else None
        if cp is not None:
            cit_prec.append(cp)
        if gr is not None:
            grounded.append(gr)

        # Answer correctness against the reference; a wrong refusal scores 0, as in SQuAD.
        f1: float | None = None
        recall: float | None = None
        if example.answerable and example.answer:
            prediction = "" if rag.refused else rag.answer
            f1 = token_f1(prediction, example.answer)
            recall = reference_recall(prediction, example.answer)
            answer_f1s.append(f1)
            if recall is not None:
                answer_recalls.append(recall)
        supported = sum(1 for c in rag.citations if c.supported)
        support_hits += supported
        support_total += len(rag.citations)

        results.append(
            QuestionResult(
                id=example.id,
                question=example.question,
                kind=example.kind,
                answerable=example.answerable,
                answer=rag.answer,
                refused=rag.refused,
                retrieved_doc_slugs=ranked_docs[:5],
                relevant_doc_ids=example.relevant_doc_ids,
                reciprocal_rank=rr,
                citation_precision=cp,
                groundedness=gr,
                answer_f1=f1,
                answer_recall=recall,
                citations_supported=supported,
                citations_total=len(rag.citations),
                timings_ms=rag.timings_ms,
            )
        )

    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    answer_metrics = {
        "citation_precision": mean(cit_prec),
        "groundedness": mean(grounded),
        "citation_support": support_hits / support_total if support_total else 0.0,
        "answer_f1": mean(answer_f1s),
        "answer_recall": mean(answer_recalls),
        "refusal_accuracy": mean([1.0 if r.refused else 0.0 for r in unanswerable]),
        "false_refusal_rate": mean([1.0 if r.refused else 0.0 for r in answerable]),
    }
    latency: dict[str, float] = {}
    for stage, values in stage_times.items():
        latency[f"{stage}_p50"] = percentile(values, 50)
        latency[f"{stage}_p95"] = percentile(values, 95)

    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({r.kind for r in results}):
        group = [r for r in results if r.kind == kind]
        by_kind[kind] = {
            "n": float(len(group)),
            "hit@5": mean(
                [
                    1.0 if set(r.retrieved_doc_slugs[:5]) & set(r.relevant_doc_ids) else 0.0
                    for r in group
                    if r.answerable
                ]
            ),
            "mrr": mean([r.reciprocal_rank for r in group if r.answerable]),
            "refused": mean([1.0 if r.refused else 0.0 for r in group]),
        }

    return EvalReport(
        config=config_name or pipeline.retriever_name,
        num_questions=len(results),
        num_answerable=len(answerable),
        num_unanswerable=len(unanswerable),
        retrieval={name: mean(vals) for name, vals in retrieval_scores.items()},
        answer=answer_metrics,
        latency_ms=latency,
        by_kind=by_kind,
        questions=results,
    )
