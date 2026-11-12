"""Evaluation runner: retrieval metrics, refusal tuning, answer metrics, ablation, significance.

Protocol (kept deliberately simple so it can be explained in an interview):

1. Retrieval has no learned parameters, so retrieval metrics use every answerable hand-labelled
   question (lexical + paraphrase) plus a synthetic set reported separately.
2. The refusal threshold *is* tuned, so it is chosen on the ``dev`` split only and every answer
   metric (refusal accuracy, citation precision, groundedness, F1) is reported on ``test``.
3. Differences between retrievers come with a paired bootstrap 95% confidence interval.
"""

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from raglens.config import Settings
from raglens.embed.factory import build_embedder, embedder_label
from raglens.eval.dataset import QAItem
from raglens.eval.metrics import (
    citation_precision,
    doc_ranking,
    groundedness,
    hit_at_k,
    ndcg_at_k,
    paired_bootstrap_ci,
    percentile,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)
from raglens.eval.synthetic import generate_synthetic
from raglens.ingest.models import Chunk
from raglens.llm.base import BaseLLM
from raglens.llm.fake import FakeLLM
from raglens.pipeline.builder import build_retriever, load_chunks
from raglens.pipeline.rag import RAGPipeline, RetrieverMode, evidence_score
from raglens.rerank.cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)

KS = (1, 3, 5, 10)
CANDIDATE_DEPTH = 30  # chunks retrieved before collapsing to a document ranking
SYSTEMS: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid+rerank")


@dataclass
class EvalConfig:
    """What to evaluate."""

    embedders: list[str] = field(default_factory=lambda: ["lsa", "hash"])
    chunk_sizes: list[int] = field(default_factory=lambda: [256, 512, 1024])
    main_embedder: str = "lsa"
    main_chunk_size: int = 512
    main_mode: RetrieverMode = "hybrid"
    top_k: int = 5
    synthetic_n: int = 40


def retrieval_records(
    pipeline: RAGPipeline,
    items: list[QAItem],
    system: str,
) -> list[dict[str, Any]]:
    """Per-question retrieval metrics for one system (``bm25``, ``dense``, ``hybrid[+rerank]``)."""
    rerank = system.endswith("+rerank")
    mode: RetrieverMode = "hybrid" if rerank else system  # type: ignore[assignment]
    rows: list[dict[str, Any]] = []
    for item in items:
        if not item.answerable:
            continue
        t0 = time.perf_counter()
        results = pipeline.retrieve(item.question, mode=mode, top_k=CANDIDATE_DEPTH)
        if rerank and pipeline.reranker is not None:
            results = pipeline.reranker.rerank(item.question, results, top_k=CANDIDATE_DEPTH)
        latency = (time.perf_counter() - t0) * 1000
        ranking = doc_ranking(results)
        relevant = set(item.relevant_doc_ids)
        row: dict[str, Any] = {
            "id": item.id,
            "type": item.type,
            "system": system,
            "question": item.question,
            "relevant": sorted(relevant),
            "top3": ranking[:3],
            "mrr": reciprocal_rank(ranking, relevant, k=10),
            "latency_ms": latency,
        }
        for k in KS:
            row[f"hit@{k}"] = hit_at_k(ranking, relevant, k)
            row[f"recall@{k}"] = recall_at_k(ranking, relevant, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranking, relevant, k)
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of every metric column plus latency percentiles."""
    if not rows:
        return {}
    metric_keys = [k for k in rows[0] if k == "mrr" or "@" in k]
    out = {k: statistics.fmean(r[k] for r in rows) for k in metric_keys}
    latencies = [r["latency_ms"] for r in rows]
    out["p50_ms"] = percentile(latencies, 50)
    out["p95_ms"] = percentile(latencies, 95)
    out["n"] = float(len(rows))
    return out


def tune_refusal_threshold(
    pipeline: RAGPipeline, dev_items: list[QAItem], mode: RetrieverMode, top_k: int
) -> tuple[float, list[dict[str, float]]]:
    """Pick the threshold that maximises balanced refusal accuracy on the dev split.

    Balanced accuracy = mean of (share of unanswerable questions refused) and (share of
    answerable questions answered), so the ten unanswerable questions count as much as the
    forty answerable ones. Ties are broken by taking the middle of the best plateau, which
    leaves the widest margin on both sides.
    """
    scores = [
        (
            item.answerable,
            evidence_score(item.question, pipeline.retrieve(item.question, mode, top_k)),
        )
        for item in dev_items
    ]
    n_ans = sum(1 for a, _ in scores if a) or 1
    n_unans = sum(1 for a, _ in scores if not a) or 1
    sweep: list[dict[str, float]] = []
    for step in range(0, 21):
        t = step / 20
        answered_ok = sum(1 for a, s in scores if a and s >= t) / n_ans
        refused_ok = sum(1 for a, s in scores if not a and s < t) / n_unans
        sweep.append(
            {
                "threshold": t,
                "answer_rate_answerable": answered_ok,
                "refusal_rate_unanswerable": refused_ok,
                "balanced_accuracy": (answered_ok + refused_ok) / 2,
            }
        )
    best = max(row["balanced_accuracy"] for row in sweep)
    plateau = [row["threshold"] for row in sweep if row["balanced_accuracy"] == best]
    return plateau[len(plateau) // 2], sweep


def answer_records(
    pipeline: RAGPipeline, items: list[QAItem], mode: RetrieverMode, top_k: int
) -> list[dict[str, Any]]:
    """Run the full pipeline and score each answer."""
    rows: list[dict[str, Any]] = []
    for item in items:
        result = pipeline.ask(item.question, mode=mode, top_k=top_k)
        relevant = set(item.relevant_doc_ids)
        by_number = {c.number: c for c in result.contexts}
        cited = [by_number[c.number] for c in result.citations]
        row: dict[str, Any] = {
            "id": item.id,
            "type": item.type,
            "answerable": item.answerable,
            "refused": result.refused,
            "correct_refusal_decision": result.refused != item.answerable,
            "evidence_score": result.evidence_score,
            "answer": result.answer,
            "citation_precision": None,
            "groundedness": None,
            "f1": None,
            "timings_ms": result.timings_ms,
        }
        if item.answerable and not result.refused:
            row["citation_precision"] = citation_precision(
                result.answer, [c.source for c in cited], relevant
            )
            row["groundedness"] = groundedness(result.answer, [c.text for c in cited])
            row["f1"] = token_f1(result.answer, item.answer)
        rows.append(row)
    return rows


def summarise_answers(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Headline answer metrics (means skip questions where a metric is undefined)."""

    def mean_of(key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.fmean(vals) if vals else 0.0

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    out = {
        "n": float(len(rows)),
        "refusal_accuracy_unanswerable": (
            statistics.fmean(1.0 if r["refused"] else 0.0 for r in unanswerable)
            if unanswerable
            else 0.0
        ),
        "false_refusal_rate_answerable": (
            statistics.fmean(1.0 if r["refused"] else 0.0 for r in answerable)
            if answerable
            else 0.0
        ),
        "decision_accuracy": statistics.fmean(
            1.0 if r["correct_refusal_decision"] else 0.0 for r in rows
        ),
        "citation_precision": mean_of("citation_precision"),
        "groundedness": mean_of("groundedness"),
        "answer_f1": mean_of("f1"),
    }
    for stage in ("retrieve", "generate", "total"):
        vals = [r["timings_ms"].get(stage, 0.0) for r in rows]
        out[f"{stage}_p50_ms"] = percentile(vals, 50)
        out[f"{stage}_p95_ms"] = percentile(vals, 95)
    return out


def _try_reranker(model_name: str) -> tuple[CrossEncoderReranker | None, str]:
    reranker = CrossEncoderReranker(model_name, fallback_pass_through=False)
    try:
        reranker._load_model()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    return reranker, "loaded"


def run_evaluation(
    items: list[QAItem],
    settings: Settings,
    config: EvalConfig,
    llm: BaseLLM | None = None,
) -> dict[str, Any]:
    """Run the whole protocol and return a JSON-serialisable result dictionary."""
    llm = llm or FakeLLM()
    reranker, reranker_status = _try_reranker(settings.reranker_model)
    chunks_cache: dict[int, list[Chunk]] = {}

    def chunks_for(size: int) -> list[Chunk]:
        if size not in chunks_cache:
            overlap = min(settings.chunk_overlap, size // 4)
            chunks_cache[size] = load_chunks(settings.corpus_dir, size, overlap)
        return chunks_cache[size]

    systems = [s for s in SYSTEMS if s != "hybrid+rerank" or reranker is not None]
    ablation: list[dict[str, Any]] = []
    main: dict[str, Any] = {}
    for emb_name in config.embedders:
        for size in config.chunk_sizes:
            embedder = build_embedder(emb_name, hash_dim=settings.hash_embed_dim)
            t0 = time.perf_counter()
            retriever = build_retriever(chunks_for(size), embedder, settings)
            index_s = time.perf_counter() - t0
            pipeline = RAGPipeline(retriever, llm, reranker=reranker)
            for system in systems:
                rows = retrieval_records(pipeline, items, system)
                agg = aggregate(rows)
                ablation.append(
                    {
                        "embedder": embedder_label(embedder),
                        "chunk_size": size,
                        "system": system,
                        "num_chunks": len(retriever.chunks),
                        "index_seconds": index_s,
                        **{
                            k: agg[k]
                            for k in ("mrr", "hit@1", "hit@5", "recall@5", "ndcg@10", "p50_ms")
                        },
                    }
                )
            if emb_name == config.main_embedder and size == config.main_chunk_size:
                main = _main_run(pipeline, items, systems, config, chunks_for(size))
                main["embedder"] = embedder_label(embedder)
                main["chunk_size"] = size
                main["num_chunks"] = len(retriever.chunks)

    return {
        "config": {
            "embedders": config.embedders,
            "chunk_sizes": config.chunk_sizes,
            "main_embedder": config.main_embedder,
            "main_chunk_size": config.main_chunk_size,
            "main_mode": config.main_mode,
            "top_k": config.top_k,
            "llm": llm.name,
            "reranker": settings.reranker_model,
            "reranker_status": reranker_status,
            "dataset_counts": {
                t: sum(1 for i in items if i.type == t) for t in sorted({i.type for i in items})
            },
        },
        "main": main,
        "ablation": ablation,
    }


def _main_run(
    pipeline: RAGPipeline,
    items: list[QAItem],
    systems: list[str],
    config: EvalConfig,
    chunks: list[Chunk],
) -> dict[str, Any]:
    synthetic = generate_synthetic(chunks, n=config.synthetic_n)
    per_system: dict[str, Any] = {}
    per_query: dict[str, list[dict[str, Any]]] = {}
    for system in systems:
        rows = retrieval_records(pipeline, items, system)
        per_query[system] = rows
        per_system[system] = {
            "all": aggregate(rows),
            "lexical": aggregate([r for r in rows if r["type"] == "lexical"]),
            "paraphrase": aggregate([r for r in rows if r["type"] == "paraphrase"]),
            "synthetic": aggregate(retrieval_records(pipeline, synthetic, system)),
        }

    significance = {}
    hybrid_mrr = [r["mrr"] for r in per_query["hybrid"]]
    for other in ("bm25", "dense"):
        mean, lo, hi = paired_bootstrap_ci(hybrid_mrr, [r["mrr"] for r in per_query[other]])
        significance[f"hybrid_vs_{other}_mrr"] = {
            "mean_diff": mean,
            "ci95_low": lo,
            "ci95_high": hi,
        }

    dev = [i for i in items if i.split == "dev"]
    test = [i for i in items if i.split == "test"]
    threshold, sweep = tune_refusal_threshold(pipeline, dev, config.main_mode, config.top_k)
    pipeline.refusal_threshold = threshold
    answers = answer_records(pipeline, test, config.main_mode, config.top_k)

    # Questions where the retrievers disagree are the most useful to read.
    disagreements = []
    for b, d, h in zip(per_query["bm25"], per_query["dense"], per_query["hybrid"], strict=True):
        if len({round(b["mrr"], 3), round(d["mrr"], 3), round(h["mrr"], 3)}) > 1:
            disagreements.append(
                {
                    "id": h["id"],
                    "type": h["type"],
                    "question": h["question"],
                    "relevant": h["relevant"],
                    "bm25_mrr": b["mrr"],
                    "dense_mrr": d["mrr"],
                    "hybrid_mrr": h["mrr"],
                    "bm25_top": b["top3"][0] if b["top3"] else "",
                    "dense_top": d["top3"][0] if d["top3"] else "",
                }
            )

    return {
        "retrieval": per_system,
        "significance": significance,
        "refusal_threshold": threshold,
        "threshold_sweep": sweep,
        "answers": summarise_answers(answers),
        "answer_rows": answers,
        "disagreements": disagreements,
        "failures": [r for r in per_query["hybrid"] if r["hit@5"] == 0.0],
        "synthetic_n": len(synthetic),
    }
