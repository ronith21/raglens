"""FastAPI service: /health, /ask, /ingest, /eval (OpenAPI docs at /docs)."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from raglens import __version__
from raglens.chunking.splitter import RecursiveCharacterChunker
from raglens.config import Settings
from raglens.embed.factory import embedder_label
from raglens.eval.dataset import load_dataset
from raglens.eval.runner import aggregate, answer_records, retrieval_records, summarise_answers
from raglens.ingest.parser import ingest_file
from raglens.llm.base import LLMError
from raglens.pipeline.builder import build_pipeline
from raglens.pipeline.rag import RAGAnswer, RAGPipeline

logger = logging.getLogger(__name__)
ALLOWED_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".pdf"}


class AskRequest(BaseModel):
    """Body of ``POST /ask``. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=1000)
    retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool = False


class EvalRequest(BaseModel):
    """Body of ``POST /eval``."""

    model_config = ConfigDict(extra="forbid")

    retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"
    split: Literal["dev", "test", "all"] = "test"


class IngestResponse(BaseModel):
    """Result of ``POST /ingest``."""

    files: list[str]
    chunks_added: int
    total_chunks: int


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    """Build the app. Pass a ready pipeline in tests to avoid loading models."""
    settings = settings or Settings()
    app = FastAPI(
        title="RAGLens",
        version=__version__,
        description="Hybrid-retrieval RAG with citations, refusals and a built-in evaluation.",
    )
    state: dict[str, RAGPipeline] = {}

    def get_pipeline() -> RAGPipeline:
        if "pipeline" not in state:
            state["pipeline"] = pipeline or build_pipeline(settings)
        return state["pipeline"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness plus what the service is running with."""
        p = get_pipeline()
        return {
            "status": "ok",
            "version": __version__,
            "chunks": len(p.retriever.chunks),
            "embedder": embedder_label(p.retriever.dense_retriever.embedder),
            "llm": p.llm.name,
            "reranker": p.reranker is not None,
            "refusal_threshold": p.refusal_threshold,
        }

    @app.post("/ask", response_model=RAGAnswer)
    def ask(body: AskRequest) -> RAGAnswer:
        """Answer a question with citations, or refuse when evidence is weak."""
        try:
            return get_pipeline().ask(
                body.question, mode=body.retriever, top_k=body.top_k, rerank=body.rerank
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMError as exc:
            logger.error("LLM failure: %s", exc)
            raise HTTPException(
                status_code=503, detail="The language model is unavailable"
            ) from exc

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
        """Add .md, .txt or .pdf files to the index (kept in memory until restart)."""
        p = get_pipeline()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        chunker = RecursiveCharacterChunker(settings.chunk_size, settings.chunk_overlap)
        new_chunks = []
        names: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            for upload in files:
                # Only the base name is used, so "../../etc/passwd" cannot escape the temp dir.
                name = Path(upload.filename or "").name
                if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                    raise HTTPException(400, f"Unsupported file type: {name or '(no name)'}")
                data = await upload.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise HTTPException(413, f"{name} exceeds {settings.max_upload_mb} MB")
                path = Path(tmp) / name
                path.write_bytes(data)
                try:
                    doc = ingest_file(path)
                except Exception as exc:
                    raise HTTPException(
                        400, f"Could not parse {name}: {type(exc).__name__}"
                    ) from exc
                new_chunks.extend(chunker.chunk_document(doc))
                names.append(name)
        if not new_chunks:
            raise HTTPException(400, "No text could be extracted from the uploaded files")
        p.retriever.index_chunks(p.retriever.chunks + new_chunks)
        return IngestResponse(
            files=names, chunks_added=len(new_chunks), total_chunks=len(p.retriever.chunks)
        )

    @app.post("/eval")
    def evaluate(body: EvalRequest) -> dict[str, Any]:
        """Score the running index on the bundled question set (FakeLLM-independent metrics)."""
        p = get_pipeline()
        dataset = Path(settings.eval_dataset)
        if not dataset.exists():
            raise HTTPException(404, f"Evaluation set not found: {dataset}")
        items = load_dataset(dataset, split=None if body.split == "all" else body.split)
        retrieval = aggregate(retrieval_records(p, items, body.retriever))
        answers = summarise_answers(
            answer_records(p, items, body.retriever, settings.default_top_k)
        )
        return {
            "retriever": body.retriever,
            "split": body.split,
            "questions": len(items),
            "retrieval": {k: round(v, 4) for k, v in retrieval.items()},
            "answers": {k: round(v, 4) for k, v in answers.items()},
        }

    return app


app = create_app()
