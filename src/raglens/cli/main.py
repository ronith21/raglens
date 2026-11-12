"""Command line interface: ``raglens ingest | ask | eval | serve``."""

import json
import logging
from pathlib import Path

import typer

from raglens.config import Settings
from raglens.embed.factory import build_embedder, embedder_label
from raglens.eval.dataset import load_dataset
from raglens.eval.report import write_report
from raglens.eval.runner import EvalConfig, run_evaluation
from raglens.pipeline.builder import build_pipeline, build_retriever, load_chunks
from raglens.pipeline.rag import RETRIEVER_MODES
from raglens.retrieve.store import IndexStore

app = typer.Typer(help="RAGLens: hybrid-retrieval RAG with a built-in evaluation harness.")


def _settings(embedder: str | None) -> Settings:
    settings = Settings()
    if embedder:
        settings.embedder = embedder
    return settings


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Show info logs")) -> None:
    """Configure logging for every command."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@app.command()
def ingest(
    corpus: Path = typer.Option(None, help="Folder of .md/.txt/.pdf files (default: CORPUS_DIR)"),
    index_dir: Path = typer.Option(None, help="Where to write the index (default: INDEX_DIR)"),
    embedder: str = typer.Option(None, help="minilm | glove | lsa | hash"),
) -> None:
    """Chunk, embed and index a corpus, then save the index to disk."""
    settings = _settings(embedder)
    corpus_dir = corpus or Path(settings.corpus_dir)
    out_dir = index_dir or Path(settings.index_dir)
    emb = build_embedder(
        settings.embedder,
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        hash_dim=settings.hash_embed_dim,
        lsa_components=settings.lsa_components,
    )
    chunks = load_chunks(corpus_dir, settings.chunk_size, settings.chunk_overlap)
    retriever = build_retriever(chunks, emb, settings)
    IndexStore.save(retriever, out_dir)
    sources = {c.metadata.get("source") for c in chunks}
    typer.echo(
        f"Indexed {len(sources)} documents as {len(chunks)} chunks with "
        f"embedder={embedder_label(emb)} -> {out_dir}"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer"),
    retriever: str = typer.Option("hybrid", help="bm25 | dense | hybrid"),
    top_k: int = typer.Option(5, min=1, max=20),
    embedder: str = typer.Option(None, help="minilm | glove | lsa | hash"),
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON result"),
) -> None:
    """Answer a question with citations (or refuse)."""
    if retriever not in RETRIEVER_MODES:
        raise typer.BadParameter(f"retriever must be one of {RETRIEVER_MODES}")
    pipeline = build_pipeline(_settings(embedder))
    result = pipeline.ask(question, mode=retriever, top_k=top_k)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"\n{result.answer}\n")
    if result.refused:
        typer.echo(f"(refused: {result.refusal_reason})")
    for c in result.contexts:
        mark = "*" if any(cit.number == c.number for cit in result.citations) else " "
        typer.echo(f" {mark}[{c.number}] {c.source}  score={c.score:.4f}")
    typer.echo(
        f"\nevidence={result.evidence_score:.2f}  timings(ms)={json.dumps(result.timings_ms)}"
    )


@app.command(name="eval")
def evaluate(
    dataset: Path = typer.Option(None, help="JSONL question set (default: EVAL_DATASET)"),
    out_dir: Path = typer.Option(None, help="Report folder (default: EVAL_OUTPUT_DIR)"),
    embedders: str = typer.Option("glove,lsa,hash", help="Comma-separated embedders to ablate"),
    main_embedder: str = typer.Option("glove", help="Embedder for the headline numbers"),
    chunk_sizes: str = typer.Option("256,512,1024", help="Comma-separated chunk sizes"),
    main_chunk_size: int = typer.Option(512),
) -> None:
    """Run the full evaluation (retrieval, refusals, answers, ablation) and write the report."""
    settings = Settings()
    items = load_dataset(dataset or Path(settings.eval_dataset))
    config = EvalConfig(
        embedders=[e.strip() for e in embedders.split(",") if e.strip()],
        chunk_sizes=[int(s) for s in chunk_sizes.split(",") if s.strip()],
        main_embedder=main_embedder,
        main_chunk_size=main_chunk_size,
        top_k=settings.default_top_k,
    )
    if config.main_embedder not in config.embedders:
        raise typer.BadParameter("main_embedder must be one of --embedders")
    if config.main_chunk_size not in config.chunk_sizes:
        raise typer.BadParameter("main_chunk_size must be one of --chunk-sizes")
    results = run_evaluation(items, settings, config)
    report = write_report(results, out_dir or Path(settings.eval_output_dir))
    main = results["main"]
    for system, scores in main["retrieval"].items():
        a = scores["all"]
        typer.echo(
            f"{system:>14}: MRR={a['mrr']:.3f} Hit@5={a['hit@5']:.3f} nDCG@10={a['ndcg@10']:.3f}"
        )
    ans = main["answers"]
    typer.echo(
        f"refusal threshold={main['refusal_threshold']:.2f}  "
        f"refused unanswerable={ans['refusal_accuracy_unanswerable']:.2f}  "
        f"false refusals={ans['false_refusal_rate_answerable']:.2f}  "
        f"citation precision={ans['citation_precision']:.2f}"
    )
    typer.echo(f"Report written to {report}")


@app.command()
def serve(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "raglens.api.main:app", host=host or settings.api_host, port=port or settings.api_port
    )


if __name__ == "__main__":
    app()
