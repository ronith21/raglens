"""Streamlit UI: ask questions, inspect evidence, and browse the evaluation report.

Run with ``streamlit run src/raglens/ui/app.py``.
"""

import json
from pathlib import Path
from typing import Any

import streamlit as st

from raglens.config import Settings
from raglens.embed.factory import embedder_label
from raglens.pipeline.builder import build_pipeline
from raglens.pipeline.rag import RAGPipeline

st.set_page_config(page_title="RAGLens", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading index and models...")
def get_pipeline() -> RAGPipeline:
    """Build the pipeline once per server process."""
    return build_pipeline(Settings())


def _highlight_sources(contexts: list[Any], cited: set[int]) -> None:
    for c in contexts:
        badge = "✅ cited" if c.number in cited else "not cited"
        signals = ", ".join(
            f"{k.replace('_', ' ')} {v:.3g}" for k, v in c.signals.items() if "rank" in k
        )
        with st.expander(f"[{c.number}] {c.title} · score {c.score:.4f} · {badge}"):
            st.caption(f"{c.source} · page {c.page} · {signals or 'single retriever'}")
            st.text(c.text)


def ask_tab(pipeline: RAGPipeline) -> None:
    """Question box, answer with citations, and the retrieved passages."""
    col_q, col_opts = st.columns([3, 1])
    with col_opts:
        mode = st.radio("Retriever", ["hybrid", "bm25", "dense"], horizontal=False)
        top_k = st.slider("Passages", 1, 10, 5)
    with col_q:
        question = st.text_input(
            "Ask a question about the corpus",
            value="How does reciprocal rank fusion combine BM25 and dense rankings?",
        )
        go = st.button("Ask", type="primary")
    if not (go and question.strip()):
        return
    result = pipeline.ask(question, mode=mode, top_k=top_k)  # type: ignore[arg-type]
    if result.refused:
        st.warning(f"**{result.answer}**  \nReason: {result.refusal_reason}")
    else:
        st.success(result.answer)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Evidence score",
        f"{result.evidence_score:.2f}",
        help="Share of question words in best passage",
    )
    m2.metric("Threshold", f"{pipeline.refusal_threshold:.2f}")
    m3.metric("Citations", len(result.citations))
    m4.metric("Latency", f"{result.timings_ms.get('total', 0):.1f} ms")
    st.subheader("Retrieved passages")
    _highlight_sources(result.contexts, {c.number for c in result.citations})


def eval_tab(report_dir: Path) -> None:
    """Show the numbers and charts produced by ``raglens eval``."""
    results_file = report_dir / "eval_results.json"
    if not results_file.exists():
        st.info("No evaluation yet. Run `raglens eval` to generate docs/proof/eval_report.md.")
        return
    results = json.loads(results_file.read_text(encoding="utf-8"))
    main = results["main"]
    st.caption(
        f"Embedder {main['embedder']}, chunk size {main['chunk_size']}, "
        f"{int(main['retrieval']['hybrid']['all']['n'])} answerable labelled questions"
    )
    rows = []
    for system, scores in main["retrieval"].items():
        a = scores["all"]
        rows.append(
            {
                "retriever": system,
                "Hit@1": a["hit@1"],
                "Hit@5": a["hit@5"],
                "MRR@10": a["mrr"],
                "nDCG@10": a["ndcg@10"],
                "paraphrase MRR": scores["paraphrase"]["mrr"],
                "p50 ms": a["p50_ms"],
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    ans = main["answers"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unanswerable refused", f"{ans['refusal_accuracy_unanswerable']:.0%}")
    c2.metric("False refusals", f"{ans['false_refusal_rate_answerable']:.0%}")
    c3.metric("Citation precision", f"{ans['citation_precision']:.0%}")
    c4.metric("Groundedness", f"{ans['groundedness']:.0%}")
    for chart in ("mrr_by_type.png", "ablation_ndcg.png", "refusal_threshold.png"):
        path = report_dir / chart
        if path.exists():
            st.image(str(path))


def main() -> None:
    """Page layout."""
    settings = Settings()
    pipeline = get_pipeline()
    st.title("🔎 RAGLens")
    st.caption(
        f"{len(pipeline.retriever.chunks)} chunks · embedder "
        f"{embedder_label(pipeline.retriever.dense_retriever.embedder)} · LLM {pipeline.llm.name}"
    )
    tab_ask, tab_eval = st.tabs(["Ask", "Evaluation"])
    with tab_ask:
        ask_tab(pipeline)
    with tab_eval:
        eval_tab(Path(settings.eval_output_dir))


main()
