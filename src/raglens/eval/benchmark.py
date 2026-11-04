"""Ablation benchmark: retriever x chunk size, written to docs/proof/.

Usage::

    python -m raglens.eval.benchmark [--dataset data/eval/qa.jsonl] [--out docs/proof]

Answers are generated with the deterministic FakeLLM so the numbers are reproducible and need
no network. The embedder follows the settings (SentenceTransformers when installed, otherwise
the HashEmbedder fallback) and is named in the report.
"""

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from raglens.chunking.splitter import RecursiveCharacterChunker
from raglens.config import Settings
from raglens.embed.base import BaseEmbedder
from raglens.engine import create_embedder, load_gate
from raglens.eval.charts import grouped_bar_chart, heatmap
from raglens.eval.dataset import QAExample, load_dataset
from raglens.eval.runner import DEFAULT_KS, EvalReport, evaluate
from raglens.ingest.parser import ingest_directory
from raglens.llm import BaseLLM, LLMError, create_llm
from raglens.llm.fake import FakeLLM
from raglens.pipeline.rag import RAGPipeline
from raglens.rerank.cross_encoder import CrossEncoderReranker
from raglens.retrieve.base import BaseRetriever
from raglens.retrieve.hybrid import HybridRetriever
from raglens.retrieve.multi_query import MultiQueryRetriever, QueryRewriter

RETRIEVER_CONFIGS = ("bm25", "dense", "hybrid", "hybrid+mq", "hybrid+rerank")
CHUNK_SIZES = (256, 512, 1024)


def _overlap_for(size: int) -> int:
    """12.5% overlap, the same ratio as the 512/64 default."""
    return size // 8


def run_ablation(
    settings: Settings,
    dataset: list[QAExample],
    embedder: BaseEmbedder,
    chunk_sizes: tuple[int, ...] = CHUNK_SIZES,
    configs: tuple[str, ...] = RETRIEVER_CONFIGS,
    llm: BaseLLM | None = None,
) -> tuple[list[EvalReport], dict[str, object]]:
    """Evaluate every (config, chunk size) pair; returns reports and run metadata.

    ``llm`` defaults to the deterministic FakeLLM; pass a real provider to measure it.
    """
    unknown = set(configs) - set(RETRIEVER_CONFIGS)
    if unknown:
        raise ValueError(f"Unknown configs {sorted(unknown)}; expected {RETRIEVER_CONFIGS}")
    documents = ingest_directory(settings.corpus_dir)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model)
    reranker_available = reranker.available if "hybrid+rerank" in configs else False
    llm = llm or FakeLLM()
    gate = load_gate(settings, embedder.name)

    reports: list[EvalReport] = []
    chunk_counts: dict[int, int] = {}
    for size in chunk_sizes:
        chunks = RecursiveCharacterChunker(size, _overlap_for(size)).chunk_documents(documents)
        chunk_counts[size] = len(chunks)
        hybrid = HybridRetriever(
            embedder=embedder,
            chunks=chunks,
            rrf_k=settings.rrf_k,
            bm25_k1=settings.bm25_k1,
            bm25_b=settings.bm25_b,
        )
        retrievers: dict[str, BaseRetriever] = {
            "bm25": hybrid.bm25_retriever,
            "dense": hybrid.dense_retriever,
            "hybrid": hybrid,
            "hybrid+mq": MultiQueryRetriever(
                hybrid, QueryRewriter.from_chunks(chunks), rrf_k=settings.rrf_k
            ),
            "hybrid+rerank": hybrid,
        }
        for config in configs:
            pipeline = RAGPipeline(
                retriever=retrievers[config],
                llm=llm,
                reranker=reranker if config == "hybrid+rerank" else None,
                retriever_name=config,
                top_k=settings.default_top_k,
                rerank_candidates=settings.rerank_candidates,
                min_score=settings.min_relevance_threshold,
                min_query_coverage=settings.min_query_coverage,
                gate=gate,
                embedder=embedder,
            )
            reports.append(evaluate(pipeline, dataset, DEFAULT_KS, f"{config}@{size}"))

    meta: dict[str, object] = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "embedder": embedder.name,
        "reranker": settings.reranker_model if reranker_available else "unavailable (pass-through)",
        "llm": _describe_llm(llm, settings),
        "corpus": settings.corpus_dir,
        "gate": type(gate).__name__,
        "documents": len(documents),
        "chunk_counts": chunk_counts,
        "questions": len(dataset),
        "answerable": sum(1 for q in dataset if q.answerable),
        "unanswerable": sum(1 for q in dataset if not q.answerable),
        "min_relevance_threshold": settings.min_relevance_threshold,
        "min_query_coverage": settings.min_query_coverage,
    }
    return reports, meta


def _describe_llm(llm: BaseLLM, settings: Settings) -> str:
    if isinstance(llm, FakeLLM):
        return "FakeLLM (deterministic, extractive)"
    model = settings.ollama_model if llm.name == "ollama" else settings.openai_model
    return f"{llm.name} ({model})"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(reports: list[EvalReport], meta: dict[str, object], elapsed_s: float) -> str:
    """Render the ablation results as a markdown report."""
    lines = [
        "# RAGLens evaluation report",
        "",
        "Generated by `python -m raglens.eval.benchmark` "
        f"on {meta['generated_at']} in {elapsed_s:.1f}s. Every number below comes from that run.",
        "",
        "## Setup",
        "",
        f"- Corpus: `{meta['corpus']}`, {meta['documents']} documents; "
        f"chunks per size: {meta['chunk_counts']}",
        f"- Questions: {meta['questions']} ({meta['answerable']} answerable, "
        f"{meta['unanswerable']} unanswerable)",
        f"- Embedder: `{meta['embedder']}`",
        f"- Reranker: `{meta['reranker']}`",
        f"- Generator: {meta['llm']}",
        f"- Refusal gate: `{meta['gate']}` (plus best score < {meta['min_relevance_threshold']})",
        f"- Python {meta['python']} on {meta['platform']}",
        "",
    ]
    if "hash" in str(meta["embedder"]):
        lines += [
            "> **Limitation:** the SentenceTransformers model was not available in this run, so",
            "> dense retrieval used the deterministic `HashEmbedder` (a bag-of-words random",
            "> projection). Dense numbers therefore measure lexical overlap, not semantics.",
            "",
        ]
    if "unavailable" in str(meta["reranker"]):
        lines += [
            "> **Limitation:** the cross-encoder could not be loaded, so `hybrid+rerank` ran in",
            "> pass-through mode. Its numbers differ from `hybrid` only because the rerank path",
            "> asks for a larger candidate pool, which changes the RRF fusion slightly.",
            "",
        ]
    if str(meta["llm"]).startswith("FakeLLM"):
        lines += [
            "> **Read with care:** the extractive FakeLLM copies context sentences, so",
            "> groundedness is close to 1.0 by construction; answer quality of a real LLM must be",
            "> measured with `--llm ollama` (or `openai`).",
            "",
        ]

    lines += [
        "## Retrieval (document level, answerable questions)",
        "",
        "| Config | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@5 | MRR | nDCG@5 | nDCG@10 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        m = r.retrieval
        lines.append(
            f"| {r.config} | {_fmt(m['hit@1'])} | {_fmt(m['hit@3'])} | {_fmt(m['hit@5'])} | "
            f"{_fmt(m['hit@10'])} | {_fmt(m['recall@5'])} | {_fmt(m['mrr'])} | "
            f"{_fmt(m['ndcg@5'])} | {_fmt(m['ndcg@10'])} |"
        )

    lines += [
        "",
        "## Answers",
        "",
        "Answer F1 / recall compare answers with the hand-written reference answers (stemmed",
        "content words; a wrong refusal scores 0). Citation support is the share of citations",
        "whose evidence sentence covers at least half of the cited claim.",
        "",
        "| Config | Answer F1 | Answer recall | Citation precision | Citation support | "
        "Groundedness | Refusal accuracy | False refusals |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        a = r.answer
        lines.append(
            f"| {r.config} | {_fmt(a['answer_f1'])} | {_fmt(a['answer_recall'])} | "
            f"{_fmt(a['citation_precision'])} | {_fmt(a['citation_support'])} | "
            f"{_fmt(a['groundedness'])} | {_fmt(a['refusal_accuracy'])} | "
            f"{_fmt(a['false_refusal_rate'])} |"
        )

    lines += [
        "",
        "## Latency (ms per question)",
        "",
        "| Config | retrieve p50 | retrieve p95 | rerank p50 | rerank p95 | generate p50 | "
        "generate p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lat = r.latency_ms
        lines.append(
            f"| {r.config} | {lat.get('retrieve_p50', 0):.2f} | {lat.get('retrieve_p95', 0):.2f} | "
            f"{lat.get('rerank_p50', 0):.2f} | {lat.get('rerank_p95', 0):.2f} | "
            f"{lat.get('generate_p50', 0):.2f} | {lat.get('generate_p95', 0):.2f} |"
        )

    kinds = sorted({k for r in reports for k in r.by_kind})
    if len(kinds) > 1:
        lines += [
            "",
            "## By question type",
            "",
            "MRR for answerable types; share refused for every type (lower is better for",
            "answerable questions, higher is better for unanswerable ones).",
            "",
            "| Config | " + " | ".join(f"{k} MRR | {k} refused" for k in kinds) + " |",
            "|---|" + "---|---|" * len(kinds),
        ]
        for r in reports:
            cells = []
            for k in kinds:
                stats = r.by_kind.get(k, {})
                mrr = "-" if k == "unanswerable" else _fmt(stats.get("mrr", 0.0))
                cells.append(f"{mrr} | {_fmt(stats.get('refused', 0.0))}")
            lines.append(f"| {r.config} | " + " | ".join(cells) + " |")

    best = max(reports, key=lambda r: (r.retrieval["mrr"], r.retrieval["ndcg@5"]))
    lines += [
        "",
        "## Observations",
        "",
        f"- Best MRR: **{best.config}** ({_fmt(best.retrieval['mrr'])}).",
    ]
    for r in reports:
        wrong = [q for q in r.questions if q.refused == q.answerable]
        if wrong:
            detail = ", ".join(
                f"`{q.id}` ({'refused answerable' if q.answerable else 'answered unanswerable'})"
                for q in wrong
            )
            lines.append(f"- {r.config} refusal errors: {detail}.")
    lines += [
        "",
        "## Charts",
        "",
        "![Retrieval metrics at chunk size 512](retrieval_metrics_512.png)",
        "",
        "![MRR by retriever and chunk size](mrr_heatmap.png)",
        "",
        "![Answer quality at chunk size 512](answer_metrics_512.png)",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    reports: list[EvalReport], meta: dict[str, object], out_dir: Path, elapsed_s: float
) -> list[Path]:
    """Write markdown report, JSON results and PNG charts into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    report_md = out_dir / "eval_report.md"
    report_md.write_text(render_markdown(reports, meta, elapsed_s), encoding="utf-8")
    written.append(report_md)

    results_json = out_dir / "eval_results.json"
    results_json.write_text(
        json.dumps(
            {"meta": meta, "reports": [r.model_dump() for r in reports]}, indent=2, default=str
        ),
        encoding="utf-8",
    )
    written.append(results_json)

    by_config = {r.config: r for r in reports}
    configs = list(dict.fromkeys(r.config.split("@")[0] for r in reports))
    sizes = sorted({int(r.config.split("@")[1]) for r in reports})
    mid = 512 if 512 in sizes else sizes[0]

    metric_names = ["hit@1", "hit@5", "recall@5", "mrr", "ndcg@5"]
    written.append(
        grouped_bar_chart(
            metric_names,
            {c: [by_config[f"{c}@{mid}"].retrieval[m] for m in metric_names] for c in configs},
            f"Retrieval quality by retriever (chunk size {mid})",
            out_dir / f"retrieval_metrics_{mid}.png",
        )
    )
    answer_names = ["answer_f1", "answer_recall", "citation_precision", "refusal_accuracy"]
    written.append(
        grouped_bar_chart(
            [n.replace("_", " ") for n in answer_names],
            {c: [by_config[f"{c}@{mid}"].answer[m] for m in answer_names] for c in configs},
            f"Answer quality by retriever (chunk size {mid}, FakeLLM)",
            out_dir / f"answer_metrics_{mid}.png",
        )
    )
    written.append(
        heatmap(
            configs,
            [str(s) for s in sizes],
            [[by_config[f"{c}@{s}"].retrieval["mrr"] for s in sizes] for c in configs],
            "MRR by retriever and chunk size",
            out_dir / "mrr_heatmap.png",
            col_label="chunk size (characters)",
        )
    )
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m raglens.eval.benchmark``."""
    parser = argparse.ArgumentParser(description="Run the RAGLens ablation benchmark.")
    parser.add_argument("--dataset", default=None, help="JSONL eval dataset path")
    parser.add_argument("--out", default="docs/proof", help="output directory")
    parser.add_argument("--chunk-sizes", default="256,512,1024", help="comma-separated sizes")
    parser.add_argument("--corpus", default=None, help="corpus directory (default: CORPUS_DIR)")
    parser.add_argument(
        "--configs",
        default=",".join(RETRIEVER_CONFIGS),
        help="comma-separated subset of " + ", ".join(RETRIEVER_CONFIGS),
    )
    parser.add_argument(
        "--llm",
        default="fake",
        choices=["fake", "ollama", "openai"],
        help="generator to evaluate (real LLMs are slow: limit --configs/--chunk-sizes)",
    )
    parser.add_argument(
        "--hash-embedder", action="store_true", help="force the offline HashEmbedder"
    )
    parser.add_argument(
        "--gate", default=None, choices=["auto", "rule", "learned"], help="refusal gate"
    )
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.corpus:
        overrides["corpus_dir"] = args.corpus
    if args.hash_embedder:
        overrides["use_hash_embedder"] = True
    if args.gate:
        overrides["refusal_gate"] = args.gate
    settings = Settings().model_copy(update=overrides)
    dataset = load_dataset(args.dataset or settings.eval_dataset)
    sizes = tuple(int(s) for s in args.chunk_sizes.split(",") if s.strip())
    configs = tuple(c.strip() for c in args.configs.split(",") if c.strip())
    llm = None if args.llm == "fake" else create_llm(settings, args.llm)

    start = time.perf_counter()
    try:
        reports, meta = run_ablation(
            settings, dataset, create_embedder(settings), sizes, configs, llm
        )
    except (LLMError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start
    for path in write_outputs(reports, meta, Path(args.out), elapsed):
        print(f"wrote {path}")
    for r in reports:
        print(
            f"{r.config:<20} hit@5={r.retrieval['hit@5']:.3f} mrr={r.retrieval['mrr']:.3f} "
            f"ndcg@5={r.retrieval['ndcg@5']:.3f} refusal_acc={r.answer['refusal_accuracy']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
