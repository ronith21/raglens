"""FastAPI service: health, ingestion, question answering, evaluation and observability."""

import importlib.resources
import json
import logging
import secrets
import shutil
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from raglens import __version__
from raglens.config import Settings
from raglens.engine import DocumentInfo, IngestReport, RAGEngine
from raglens.eval.dataset import load_dataset
from raglens.eval.runner import DEFAULT_KS, EvalReport, evaluate
from raglens.ingest.parser import SUPPORTED_EXTENSIONS
from raglens.llm.base import LLMError
from raglens.observability import (
    PROMETHEUS_CONTENT_TYPE,
    AnalyticsSummary,
    AppMetrics,
    QueryLog,
    QueryRecord,
    configure_logging,
)
from raglens.pipeline.rag import RAGAnswer

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("raglens.access")

RetrieverName = Literal["bm25", "dense", "hybrid"]
ProviderName = Literal["fake", "ollama", "openai"]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class HealthResponse(BaseModel):
    """Service liveness and index summary."""

    status: str
    version: str
    documents: int
    chunks: int
    embedder: str
    llm_provider: str
    reranker_enabled: bool
    corpus: str = ""
    refusal_gate: str = ""


class AskRequest(BaseModel):
    """Body of ``POST /ask``."""

    question: str = Field(..., min_length=1, max_length=2000, examples=["What is BM25?"])
    retriever: RetrieverName = "hybrid"
    rerank: bool | None = Field(None, description="Override ENABLE_RERANKER for this request")
    top_k: int = Field(5, ge=1, le=20)
    llm_provider: ProviderName | None = Field(None, description="Override LLM_PROVIDER")
    multi_query: bool = Field(False, description="Fuse retrieval over rewritten query variants")


class EvalRequest(BaseModel):
    """Body of ``POST /eval``."""

    retriever: RetrieverName = "hybrid"
    rerank: bool = False
    multi_query: bool = False
    top_k: int = Field(5, ge=1, le=20)
    limit: int | None = Field(None, ge=1, description="Evaluate only the first N questions")
    include_questions: bool = Field(True, description="Return per-question results")


def get_engine(request: Request) -> RAGEngine:
    """Dependency returning the app's engine."""
    engine: RAGEngine = request.app.state.engine
    return engine


EngineDep = Annotated[RAGEngine, Depends(get_engine)]


def create_app(settings: Settings | None = None, engine: RAGEngine | None = None) -> FastAPI:
    """Build the FastAPI app. Tests pass a pre-built engine; production builds one on startup."""
    app_settings = settings or (engine.settings if engine else Settings())
    configure_logging(app_settings.log_level, app_settings.log_json)
    metrics = AppMetrics()
    query_log = QueryLog(
        app_settings.query_log_path,
        enabled=app_settings.query_log_enabled,
        redact=app_settings.redact_pii,
    )

    def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        """Reject the request unless API_KEY is unset or the X-API-Key header matches it."""
        expected = app_settings.api_key
        if expected and not secrets.compare_digest(x_api_key or "", expected):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing or invalid X-API-Key header",
                headers={"WWW-Authenticate": "API-Key"},
            )

    protected = [Depends(require_api_key)]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if engine is None:
            built = RAGEngine(app_settings)
            built.load_or_build()
            app.state.engine = built
        else:
            app.state.engine = engine
        yield

    app = FastAPI(
        title="RAGLens API",
        version=__version__,
        description=(
            "Retrieval-Augmented Generation with hybrid BM25 + dense retrieval, optional "
            "cross-encoder reranking, inline citations and a built-in evaluation harness."
        ),
        lifespan=lifespan,
        docs_url=None,  # served below from bundled assets, so /docs works offline
    )
    app.state.metrics = metrics
    app.state.query_log = query_log

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Assign a request id, time the request, record metrics and an access log line."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            template = getattr(route, "path", "unmatched")
            if template != "/metrics":
                metrics.http_requests.labels(request.method, template, str(status_code)).inc()
                metrics.http_latency.labels(template).observe(elapsed)
            access_logger.info(
                "%s %s %s",
                request.method,
                request.url.path,
                status_code,
                extra={
                    "request_id": request_id,
                    "route": template,
                    "status": status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )
        response.headers["X-Request-ID"] = request_id
        return response

    # Swagger UI assets bundled by fastapi-offline (OpenAPI 3.1 capable), so /docs needs no CDN.
    swagger_dir = Path(str(importlib.resources.files("fastapi_offline") / "static"))
    app.mount("/static/swagger", StaticFiles(directory=swagger_dir), name="swagger")

    @app.get("/docs", include_in_schema=False)
    def swagger_docs() -> HTMLResponse:
        """Swagger UI using locally bundled JS/CSS (no CDN required)."""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url or "/openapi.json",
            title=f"{app.title} - docs",
            swagger_js_url="/static/swagger/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger/swagger-ui.css",
            swagger_favicon_url="/static/swagger/favicon.png",
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(engine: EngineDep) -> HealthResponse:
        """Liveness probe with index statistics."""
        metrics.index_chunks.set(len(engine.chunks))
        return HealthResponse(
            status="ok",
            version=__version__,
            documents=len({c.doc_id for c in engine.chunks}),
            chunks=len(engine.chunks),
            embedder=engine.embedder.name,
            llm_provider=engine.settings.llm_provider,
            reranker_enabled=engine.settings.enable_reranker,
            corpus=Path(engine.settings.corpus_dir).name,
            refusal_gate=type(engine.gate).__name__,
        )

    @app.get("/documents", response_model=list[DocumentInfo], tags=["index"])
    def documents(engine: EngineDep) -> list[DocumentInfo]:
        """List indexed documents."""
        return engine.documents()

    @app.post("/ingest", response_model=IngestReport, tags=["index"], dependencies=protected)
    async def ingest(
        engine: EngineDep, files: Annotated[list[UploadFile], File(...)]
    ) -> IngestReport:
        """Upload .md / .txt / .pdf files, chunk and index them (idempotent per file)."""
        with tempfile.TemporaryDirectory() as tmp:
            saved: list[Path] = []
            for upload in files:
                name = Path(upload.filename or "").name
                if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    raise HTTPException(
                        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        f"Unsupported file type for {name!r}; allowed: "
                        f"{sorted(SUPPORTED_EXTENSIONS)}",
                    )
                data = await upload.read()
                if len(data) > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"{name!r} exceeds 20 MB"
                    )
                target = Path(tmp) / name
                target.write_bytes(data)
                saved.append(target)

            uploads_dir = Path(engine.settings.corpus_dir) / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            kept = []
            for path in saved:
                dest = uploads_dir / path.name
                shutil.copyfile(path, dest)
                kept.append(dest)
            try:
                report = engine.ingest_paths(kept)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            metrics.index_chunks.set(report.total_chunks)
            return report

    @app.post("/ask", response_model=RAGAnswer, tags=["qa"])
    def ask(body: AskRequest, request: Request, engine: EngineDep) -> RAGAnswer:
        """Answer a question with inline citations, or refuse when evidence is weak."""
        if not body.question.strip():
            raise HTTPException(422, "question must not be blank")
        if not engine.chunks:
            raise HTTPException(status.HTTP_409_CONFLICT, "The index is empty; ingest first")
        start = time.perf_counter()
        try:
            answer = engine.ask(
                body.question,
                retriever=body.retriever,
                rerank=body.rerank,
                top_k=body.top_k,
                llm_provider=body.llm_provider,
                multi_query=body.multi_query,
            )
        except LLMError as exc:
            metrics.answers.labels(body.retriever, "error").inc()
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
        record_answer(answer, request.state.request_id, time.perf_counter() - start)
        return answer

    def record_answer(answer: RAGAnswer, request_id: str, elapsed_s: float) -> None:
        """Update answer metrics and append the question to the query log."""
        metrics.answers.labels(answer.retriever, "refused" if answer.refused else "answered").inc()
        for stage, ms in answer.timings_ms.items():
            metrics.stage_latency.labels(stage).observe(ms / 1000.0)
        top = answer.contexts[0].chunk.metadata.get("title", "") if answer.contexts else ""
        query_log.append(
            QueryRecord(
                ts=datetime.now(UTC).isoformat(timespec="seconds"),
                request_id=request_id,
                question=answer.question,
                retriever=answer.retriever,
                refused=answer.refused,
                refusal_reason=answer.refusal_reason,
                top_document="" if answer.refused else str(top),
                citations=len(answer.citations),
                citations_supported=sum(1 for c in answer.citations if c.supported),
                query_coverage=answer.query_coverage,
                latency_ms=round(elapsed_s * 1000.0, 3),
                timings_ms=answer.timings_ms,
            )
        )

    @app.post("/eval", response_model=EvalReport, tags=["eval"], dependencies=protected)
    def run_eval(body: EvalRequest, engine: EngineDep) -> EvalReport:
        """Evaluate the current index on the configured dataset with the configured LLM."""
        dataset_path = Path(engine.settings.eval_dataset)
        if not dataset_path.exists():
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Dataset not found: {dataset_path}")
        dataset = load_dataset(dataset_path)
        if body.limit:
            dataset = dataset[: body.limit]
        pipeline = engine.pipeline(
            body.retriever, body.rerank, body.top_k, multi_query=body.multi_query
        )
        start = time.perf_counter()
        try:
            report = evaluate(pipeline, dataset, DEFAULT_KS)
        except LLMError as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
        logger.info("Evaluated %d questions in %.2fs", len(dataset), time.perf_counter() - start)
        if not body.include_questions:
            report.questions = []
        return report

    @app.get("/eval/latest", tags=["eval"])
    def latest_benchmark() -> dict[str, Any]:
        """Return the most recent ablation results written by the benchmark (if any)."""
        path = Path(app_settings.eval_results_path)
        if not path.exists():
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, "No benchmark results yet; run `make eval`"
            )
        result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return result

    @app.get("/metrics", tags=["system"], response_class=Response)
    def prometheus_metrics() -> Response:
        """Prometheus exposition format: requests, latency, answer outcomes, index size."""
        return Response(metrics.render(), media_type=PROMETHEUS_CONTENT_TYPE)

    @app.get("/analytics", response_model=AnalyticsSummary, tags=["system"], dependencies=protected)
    def analytics(
        limit: Annotated[int, Query(ge=1, le=100_000)] = 5000,
    ) -> AnalyticsSummary:
        """SQL aggregates over the most recent ``limit`` logged questions."""
        return query_log.summarize(window=limit)

    return app


app = create_app()
