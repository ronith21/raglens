"""Streamlit front-end for RAGLens. Run with ``streamlit run src/raglens/ui/app.py``.

The UI is a thin client: every action goes through the FastAPI service at ``API_URL``.
"""

import html
import re
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

from raglens.config import Settings

SETTINGS = Settings()
API_URL = SETTINGS.api_url.rstrip("/")
# API_URL=embedded runs the FastAPI app inside this process (single container, e.g. free hosting).
EMBEDDED = API_URL == "embedded"
HEADERS = {"X-API-Key": SETTINGS.api_key} if SETTINGS.api_key else {}
PROOF_DIR = Path(SETTINGS.eval_results_path).parent

BANKING_EXAMPLES = [
    "How much is protected per depositor if my bank fails?",
    "How is the leverage ratio defined?",
    "What are the three elements of strong customer authentication?",
    "What was Deutsche Bank's CET1 ratio in 2024?",
]
EXAMPLES = [
    "Why is reciprocal rank fusion preferred over linearly combining BM25 and cosine scores?",
    "How does AdamW differ from Adam?",
    "How do isolation forests detect anomalies?",
    "What is the capital of Australia?",
]

CSS = """
<style>
:root { --accent: #2a78d6; --ink: #0b0b0b; --muted: #52514e; --line: #e4e3df; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1 { letter-spacing: -0.02em; }
.answer-card {
  border: 1px solid var(--line); border-left: 4px solid var(--accent);
  border-radius: 10px; padding: 1.1rem 1.3rem; background: rgba(42,120,214,0.04);
  font-size: 1.05rem; line-height: 1.65;
}
.answer-card.refused { border-left-color: #eda100; background: rgba(237,161,0,0.07); }
.cite {
  display: inline-block; min-width: 1.4em; padding: 0 0.35em; margin: 0 0.1em;
  border-radius: 6px; background: var(--accent); color: #fff; font-size: 0.78em;
  font-weight: 600; text-align: center; vertical-align: 0.1em;
}
.pill {
  display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.8rem;
  border: 1px solid var(--line); margin-right: 0.35rem; color: var(--muted);
}
.pill.ok { border-color: #1baf7a; color: #0f7a54; }
.pill.down { border-color: #e34948; color: #b42f2e; }
.chunk-meta { color: var(--muted); font-size: 0.85rem; }
.evidence {
  margin: 0.15rem 0 0.7rem 1.9rem; padding: 0.45rem 0.8rem; border-left: 3px solid var(--line);
  color: var(--muted); font-size: 0.9rem; line-height: 1.5;
}
.support { font-size: 0.75rem; padding: 0.05rem 0.5rem; border-radius: 999px; margin-left: 0.4rem; }
.support.ok { background: rgba(27,175,122,0.14); color: #0f7a54; }
.support.weak { background: rgba(237,161,0,0.16); color: #8a5a00; }
.passage { white-space: pre-wrap; font-size: 0.9rem; line-height: 1.55; }
.passage mark { background: rgba(42,120,214,0.18); color: inherit; padding: 0.05rem 0.1rem;
  border-radius: 3px; }
.variants { color: var(--muted); font-size: 0.85rem; }
</style>
"""


@st.cache_resource(show_spinner="Building the index…")
def _embedded_client() -> Any:
    """An in-process client for the FastAPI app (runs its startup, i.e. loads the index)."""
    from fastapi.testclient import TestClient

    from raglens.api.main import create_app

    client = TestClient(create_app(SETTINGS))
    client.__enter__()
    return client


def api_request(method: str, path: str, timeout: float = 10.0, **kwargs: Any) -> httpx.Response:
    """Send a request to the API over HTTP, or in-process in embedded mode."""
    if EMBEDDED:
        response: httpx.Response = _embedded_client().request(
            method, path, headers=HEADERS, **kwargs
        )
        return response
    return httpx.request(method, f"{API_URL}{path}", timeout=timeout, headers=HEADERS, **kwargs)


def api_get(path: str) -> Any:
    """GET from the API, returning parsed JSON or None on failure."""
    try:
        response = api_request("GET", path)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def api_post(path: str, timeout: float = 120.0, **kwargs: Any) -> tuple[Any, str | None]:
    """POST to the API, returning (json, error message)."""
    try:
        response = api_request("POST", path, timeout=timeout, **kwargs)
    except httpx.HTTPError as exc:
        return None, f"Could not reach the API at {API_URL}: {exc}"
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"API error {response.status_code}: {detail}"
    return response.json(), None


def render_answer(text: str) -> str:
    """Escape the answer and turn [n] markers into citation badges."""
    escaped = html.escape(text)
    return re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', escaped)


def highlight_passage(text: str, spans: list[tuple[int, int, int]]) -> str:
    """Escape ``text`` and wrap each ``(start, end, citation)`` span in a <mark>."""
    out: list[str] = []
    cursor = 0
    for start, end, index in sorted(s for s in spans if 0 <= s[0] < s[1] <= len(text)):
        if start < cursor:
            continue
        out.append(html.escape(text[cursor:start]))
        out.append(f'<mark title="evidence for [{index}]">{html.escape(text[start:end])}</mark>')
        cursor = end
    out.append(html.escape(text[cursor:]))
    return f'<div class="passage">{"".join(out)}</div>'


def support_badge(support: float) -> str:
    """Green badge for supported citations, amber for weak ones."""
    kind = "ok" if support >= 0.5 else "weak"
    label = "supported" if support >= 0.5 else "weak support"
    return f'<span class="support {kind}">{label} · {support:.0%}</span>'


def sidebar() -> dict[str, Any]:
    """Status panel and query settings."""
    st.sidebar.title("RAGLens")
    health = api_get("/health")
    if health:
        st.sidebar.markdown(
            '<span class="pill ok">API online</span>'
            f'<span class="pill">v{health["version"]}</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(
            f"Corpus `{health.get('corpus', '')}` · {health['documents']} documents · "
            f"{health['chunks']} chunks  \n"
            f"Embedder: `{health['embedder']}`  \nLLM: `{health['llm_provider']}`  \n"
            f"Refusal gate: `{health.get('refusal_gate', '')}`"
            + ("  \nMode: embedded (single process)" if EMBEDDED else "")
        )
    else:
        st.sidebar.markdown('<span class="pill down">API offline</span>', unsafe_allow_html=True)
        st.sidebar.caption(f"Start it with `make serve` (expected at {API_URL}).")

    st.sidebar.divider()
    st.sidebar.subheader("Retrieval")
    options = {
        "retriever": st.sidebar.radio(
            "Retriever",
            ["hybrid", "bm25", "dense"],
            help="Hybrid fuses BM25 and dense rankings with Reciprocal Rank Fusion (k=60).",
        ),
        "rerank": st.sidebar.toggle("Cross-encoder rerank", value=False),
        "multi_query": st.sidebar.toggle(
            "Multi-query rewriting",
            value=False,
            help="Also search a keyword form and an acronym-expanded form of the question, "
            "then fuse the rankings with RRF.",
        ),
        "top_k": st.sidebar.slider("Passages (top-k)", 1, 10, 5),
        "llm_provider": st.sidebar.selectbox("LLM provider", ["fake", "ollama", "openai"]),
    }
    corpus = str((health or {}).get("corpus", ""))
    examples = BANKING_EXAMPLES if "banking" in corpus else EXAMPLES
    return options | {"online": bool(health), "examples": examples}


def ask_tab(options: dict[str, Any]) -> None:
    """Question box, answer with citations, and retrieved passages."""
    st.subheader("Ask the corpus")
    examples = options["examples"]
    if "question" not in st.session_state:
        st.session_state.question = examples[0]

    cols = st.columns(len(examples))
    for col, example in zip(cols, examples, strict=True):
        if col.button(example[:38] + "…", key=f"ex-{example}", width="stretch"):
            st.session_state.question = example

    question = st.text_area("Question", key="question", height=80)
    if not st.button("Ask", type="primary", disabled=not options["online"]):
        return

    payload = {
        "question": question,
        "retriever": options["retriever"],
        "rerank": options["rerank"],
        "top_k": options["top_k"],
        "llm_provider": options["llm_provider"],
        "multi_query": options["multi_query"],
    }
    with st.spinner("Retrieving and generating…"):
        result, error = api_post("/ask", json=payload)
    if error:
        st.error(error)
        return

    card_class = "answer-card refused" if result["refused"] else "answer-card"
    st.markdown(
        f'<div class="{card_class}">{render_answer(result["answer"])}</div>',
        unsafe_allow_html=True,
    )
    if result["refused"]:
        st.caption(f"Refused: {result['refusal_reason']}")

    timings = result["timings_ms"]
    metric_cols = st.columns(5)
    metric_cols[0].metric("Retriever", result["retriever"])
    metric_cols[1].metric("Retrieve", f"{timings.get('retrieve', 0):.1f} ms")
    metric_cols[2].metric("Rerank", f"{timings.get('rerank', 0):.1f} ms")
    metric_cols[3].metric("Generate", f"{timings.get('generate', 0):.1f} ms")
    metric_cols[4].metric("Query coverage", f"{result['query_coverage']:.0%}")
    variants = (
        result["contexts"][0]["metadata"].get("query_variants") if result["contexts"] else None
    )
    if variants:
        st.markdown(
            '<div class="variants">Searched as: '
            + " · ".join(f"<code>{html.escape(v)}</code>" for v in variants)
            + "</div>",
            unsafe_allow_html=True,
        )

    if result["citations"]:
        st.markdown("**Sources and evidence**")
        for c in result["citations"]:
            st.markdown(
                f'<span class="cite">{c["index"]}</span> {html.escape(c["title"])} '
                f'<span class="chunk-meta">· {html.escape(Path(c["source_path"]).name)}</span>'
                + support_badge(c["support"])
                + (
                    f'<div class="evidence">{html.escape(c["evidence"])}</div>'
                    if c["evidence"]
                    else ""
                ),
                unsafe_allow_html=True,
            )

    st.markdown("**Retrieved passages**")
    cited = {c["index"] for c in result["citations"]}
    evidence: dict[str, list[tuple[int, int, int]]] = {}
    for c in result["citations"]:
        evidence.setdefault(c["chunk_id"], []).append(
            (c["evidence_start"], c["evidence_end"], c["index"])
        )
    for i, ctx in enumerate(result["contexts"], start=1):
        chunk = ctx["chunk"]
        marker = " ✓ cited" if i in cited else ""
        title = chunk["metadata"].get("title", chunk["doc_id"])
        with st.expander(f"[{i}] {title} — score {ctx['score']:.4f}{marker}", expanded=i == 1):
            diag = {k: v for k, v in ctx["metadata"].items() if k.endswith(("_rank", "_score"))}
            st.markdown(
                f'<div class="chunk-meta">{html.escape(chunk["chunk_id"])} · chars '
                f"{chunk['start_char']}-{chunk['end_char']} · "
                + " · ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in diag.items()
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                highlight_passage(chunk["text"], evidence.get(chunk["chunk_id"], [])),
                unsafe_allow_html=True,
            )


def eval_tab(options: dict[str, Any]) -> None:
    """Benchmark dashboard plus an on-demand live evaluation."""
    st.subheader("Evaluation dashboard")
    latest = api_get("/eval/latest")
    if latest:
        meta = latest["meta"]
        st.caption(
            f"Ablation run {meta['generated_at']} · {meta['questions']} questions "
            f"({meta['unanswerable']} unanswerable) · embedder `{meta['embedder']}` · "
            f"reranker `{meta['reranker']}`"
        )
        rows = []
        for report in latest["reports"]:
            config, size = report["config"].split("@")
            rows.append(
                {
                    "retriever": config,
                    "chunk size": int(size),
                    "Hit@1": report["retrieval"]["hit@1"],
                    "Hit@5": report["retrieval"]["hit@5"],
                    "MRR": report["retrieval"]["mrr"],
                    "nDCG@5": report["retrieval"]["ndcg@5"],
                    "answer F1": report["answer"].get("answer_f1", 0.0),
                    "citation precision": report["answer"]["citation_precision"],
                    "refusal accuracy": report["answer"]["refusal_accuracy"],
                    "false refusals": report["answer"]["false_refusal_rate"],
                    "retrieve p95 (ms)": report["latency_ms"].get("retrieve_p95", 0.0),
                }
            )
        frame = pd.DataFrame(rows)
        best = frame.sort_values("MRR", ascending=False).iloc[0]
        kpis = st.columns(4)
        kpis[0].metric(
            "Best MRR", f"{best['MRR']:.3f}", f"{best['retriever']} @ {best['chunk size']}"
        )
        kpis[1].metric("Best Hit@1", f"{frame['Hit@1'].max():.3f}")
        kpis[2].metric("Refusal accuracy (max)", f"{frame['refusal accuracy'].max():.0%}")
        kpis[3].metric("Configs evaluated", len(frame))
        st.dataframe(
            frame.style.format(precision=3).background_gradient(
                subset=["MRR", "nDCG@5"], cmap="Blues"
            ),
            width="stretch",
            hide_index=True,
        )
        chart_cols = st.columns(2)
        for col, name in zip(
            chart_cols, ["retrieval_metrics_512.png", "mrr_heatmap.png"], strict=True
        ):
            if (PROOF_DIR / name).exists():
                col.image(str(PROOF_DIR / name), width="stretch")
    else:
        st.info("No benchmark results yet. Run `make eval` to generate docs/proof/.")

    st.divider()
    st.markdown("**Live evaluation** of the current sidebar settings against the eval set")
    if st.button("Run evaluation", disabled=not options["online"]):
        payload = {
            "retriever": options["retriever"],
            "rerank": options["rerank"],
            "multi_query": options["multi_query"],
            "top_k": options["top_k"],
        }
        with st.spinner("Evaluating every question…"):
            report, error = api_post("/eval", json=payload, timeout=600.0)
        if error:
            st.error(error)
            return
        cols = st.columns(5)
        cols[0].metric("MRR", f"{report['retrieval']['mrr']:.3f}")
        cols[1].metric("Hit@5", f"{report['retrieval']['hit@5']:.3f}")
        cols[2].metric("Answer F1", f"{report['answer']['answer_f1']:.3f}")
        cols[3].metric("Citation precision", f"{report['answer']['citation_precision']:.3f}")
        cols[4].metric("Refusal accuracy", f"{report['answer']['refusal_accuracy']:.0%}")
        per_q = pd.DataFrame(
            [
                {
                    "id": q["id"],
                    "question": q["question"],
                    "answerable": q["answerable"],
                    "refused": q["refused"],
                    "RR": q["reciprocal_rank"],
                    "answer F1": q["answer_f1"],
                    "top docs": ", ".join(q["retrieved_doc_slugs"][:3]),
                }
                for q in report["questions"]
            ]
        )
        st.dataframe(per_q, width="stretch", hide_index=True)


def analytics_tab() -> None:
    """Live usage analytics from the query log, plus the raw Prometheus metrics."""
    st.subheader("Usage analytics")
    summary = api_get("/analytics")
    if not summary or not summary["total_queries"]:
        st.info("No questions logged yet. Ask something in the Ask tab, then come back.")
        return
    st.caption(f"{summary['first_ts']} → {summary['last_ts']} · from the API's query log")
    kpis = st.columns(5)
    kpis[0].metric("Questions", summary["total_queries"])
    kpis[1].metric("Refusal rate", f"{summary['refusal_rate']:.0%}")
    kpis[2].metric("Citation support", f"{summary['citation_support_rate']:.0%}")
    kpis[3].metric("Latency p50", f"{summary['latency_p50_ms']:.1f} ms")
    kpis[4].metric("Latency p95", f"{summary['latency_p95_ms']:.1f} ms")

    def counts(key: str, label: str) -> pd.DataFrame:
        return pd.DataFrame(summary[key]).rename(columns={"label": label, "count": "count"})

    left, right = st.columns(2)
    with left:
        st.markdown("**Most asked**")
        st.dataframe(counts("top_questions", "question"), width="stretch", hide_index=True)
        st.markdown("**Queries by retriever**")
        by_retriever = counts("by_retriever", "retriever")
        if not by_retriever.empty:
            st.bar_chart(by_retriever, x="retriever", y="count", height=220)
    with right:
        st.markdown("**Most-used sources**")
        st.dataframe(counts("top_documents", "document"), width="stretch", hide_index=True)
        st.markdown("**Refusal reasons**")
        st.dataframe(counts("refusal_reasons", "reason"), width="stretch", hide_index=True)

    st.markdown("**Recent questions**")
    recent = pd.DataFrame(summary["recent"])
    if not recent.empty:
        st.dataframe(
            recent[
                ["ts", "question", "retriever", "refused", "citations", "latency_ms", "request_id"]
            ],
            width="stretch",
            hide_index=True,
        )
    with st.expander("Prometheus metrics (GET /metrics)"):
        try:
            text = api_request("GET", "/metrics").text
        except httpx.HTTPError as exc:
            text = f"Could not fetch metrics: {exc}"
        st.code(
            "\n".join(line for line in text.splitlines() if line.startswith("raglens_")), "text"
        )


def documents_tab(options: dict[str, Any]) -> None:
    """Indexed documents and upload."""
    st.subheader("Documents")
    docs = api_get("/documents") or []
    if docs:
        st.dataframe(
            pd.DataFrame(docs)[["title", "num_chunks", "source_path", "doc_id"]],
            width="stretch",
            hide_index=True,
        )
    uploads = st.file_uploader(
        "Add documents (.md, .txt, .pdf)", type=["md", "txt", "pdf"], accept_multiple_files=True
    )
    if uploads and st.button("Ingest", disabled=not options["online"]):
        files = [("files", (u.name, u.getvalue())) for u in uploads]
        report, error = api_post("/ingest", files=files)
        if error:
            st.error(error)
        else:
            st.success(
                f"Added {report['documents_added']} documents ({report['chunks_added']} chunks); "
                f"index now has {report['total_chunks']} chunks."
            )


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="RAGLens", page_icon="🔎", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    options = sidebar()
    st.title("RAGLens")
    st.caption("Hybrid retrieval · inline citations · refusal gating · built-in evaluation")
    ask, evaluation, analytics, documents = st.tabs(["Ask", "Evaluation", "Analytics", "Documents"])
    with ask:
        ask_tab(options)
    with evaluation:
        eval_tab(options)
    with analytics:
        analytics_tab()
    with documents:
        documents_tab(options)


main()
