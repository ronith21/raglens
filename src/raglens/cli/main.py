"""Typer command-line interface: ``raglens ingest | ask | eval | synth | serve``."""

import json
import logging
from pathlib import Path

import typer

from raglens.config import Settings
from raglens.engine import RETRIEVERS, RAGEngine
from raglens.eval.benchmark import main as benchmark_main
from raglens.eval.dataset import load_dataset, save_dataset
from raglens.eval.runner import DEFAULT_KS, evaluate
from raglens.eval.synthetic import generate_synthetic_qa
from raglens.llm.base import LLMError
from raglens.retrieve.store import IndexStore

app = typer.Typer(help="RAGLens: retrieval-augmented generation with an evaluation harness.")


def _engine(load: bool = True) -> RAGEngine:
    engine = RAGEngine(Settings())
    if load:
        engine.load_or_build()
    return engine


def _check_retriever(name: str) -> str:
    if name not in RETRIEVERS:
        raise typer.BadParameter(f"must be one of {', '.join(RETRIEVERS)}")
    return name


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging")) -> None:
    """Configure logging for every command."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(None, help="Files or folders (default: CORPUS_DIR)"),
    rebuild: bool = typer.Option(False, help="Discard the existing index first"),
) -> None:
    """Ingest documents and (re)build the persisted index."""
    engine = _engine(load=False)
    if not rebuild and IndexStore.exists(engine.settings.index_dir):
        engine.load_or_build()  # add to the existing index instead of starting over
    targets = paths or [Path(engine.settings.corpus_dir)]
    missing = [p for p in targets if not p.exists()]
    if missing:
        typer.secho(f"Not found: {', '.join(map(str, missing))}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    report = engine.ingest_paths(targets, rebuild=rebuild)
    typer.echo(
        f"Added {report.documents_added} documents ({report.chunks_added} chunks), skipped "
        f"{report.documents_skipped}. Index: {report.total_documents} documents, "
        f"{report.total_chunks} chunks -> {engine.settings.index_dir}"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer"),
    retriever: str = typer.Option("hybrid", callback=_check_retriever, help="bm25|dense|hybrid"),
    rerank: bool = typer.Option(False, help="Apply cross-encoder reranking"),
    multi_query: bool = typer.Option(False, help="Fuse retrieval over rewritten query variants"),
    top_k: int = typer.Option(5, min=1, max=20),
    provider: str = typer.Option(None, help="LLM provider override: fake|ollama|openai"),
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON response"),
) -> None:
    """Answer a question with inline citations."""
    engine = _engine()
    try:
        answer = engine.ask(question, retriever, rerank, top_k, provider, multi_query)
    except (LLMError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(answer.model_dump_json(indent=2))
        return
    typer.secho(answer.answer, bold=True)
    if answer.refused:
        typer.secho(f"(refused: {answer.refusal_reason})", fg=typer.colors.YELLOW)
    for citation in answer.citations:
        typer.echo(f"  [{citation.index}] {citation.title} ({citation.source_path})")
        if citation.evidence:
            mark = "supported" if citation.supported else "weak support"
            typer.secho(
                f"      evidence ({mark} {citation.support:.0%}): {citation.evidence}", dim=True
            )
    timings = ", ".join(f"{k} {v:.1f} ms" for k, v in answer.timings_ms.items())
    typer.secho(f"{answer.retriever} | {answer.llm} | {timings}", dim=True)


@app.command("eval")
def eval_command(
    retriever: str = typer.Option("hybrid", callback=_check_retriever),
    rerank: bool = typer.Option(False),
    multi_query: bool = typer.Option(False),
    ablation: bool = typer.Option(
        False, help="Run the full retriever x chunk-size ablation and write docs/proof/"
    ),
    out: Path = typer.Option(Path("docs/proof"), help="Ablation output directory"),
    dataset: Path = typer.Option(None, help="JSONL dataset (default: EVAL_DATASET)"),
) -> None:
    """Evaluate one configuration, or run the ablation benchmark with --ablation."""
    if ablation:
        argv = ["--out", str(out)] + (["--dataset", str(dataset)] if dataset else [])
        raise typer.Exit(code=benchmark_main(argv))

    engine = _engine()
    examples = load_dataset(dataset or engine.settings.eval_dataset)
    report = evaluate(
        engine.pipeline(retriever, rerank, multi_query=multi_query), examples, DEFAULT_KS
    )
    summary = {
        "config": report.config,
        "questions": report.num_questions,
        "retrieval": {k: round(v, 4) for k, v in report.retrieval.items()},
        "answer": {k: round(v, 4) for k, v in report.answer.items()},
        "latency_ms": {k: round(v, 3) for k, v in report.latency_ms.items()},
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def synth(
    out: Path = typer.Option(Path("data/eval/synthetic.jsonl"), help="Output JSONL path"),
    num: int = typer.Option(20, min=1, help="Number of questions"),
    seed: int = typer.Option(13),
) -> None:
    """Generate synthetic QA pairs from the indexed chunks."""
    engine = _engine()
    examples = generate_synthetic_qa(engine.chunks, num_questions=num, seed=seed)
    save_dataset(examples, out)
    typer.echo(f"Wrote {len(examples)} synthetic questions to {out}")


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind host (default: API_HOST)"),
    port: int = typer.Option(None, help="Bind port (default: API_PORT)"),
    reload: bool = typer.Option(False),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "raglens.api.main:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
