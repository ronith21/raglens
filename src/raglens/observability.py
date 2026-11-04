"""Observability: Prometheus metrics, structured JSON logs and a SQLite query log."""

import json
import logging
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    Histogram,
    disable_created_metrics,
    generate_latest,
)
from prometheus_client import Counter as PromCounter
from pydantic import BaseModel, Field

from raglens.eval.metrics import percentile
from raglens.privacy import redact_pii

logger = logging.getLogger(__name__)

PROMETHEUS_CONTENT_TYPE = CONTENT_TYPE_LATEST
disable_created_metrics()  # type: ignore[no-untyped-call]  # drop noisy *_created series
_LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class AppMetrics:
    """Prometheus collectors on a private registry (one per app, so tests stay isolated)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http_requests = PromCounter(
            "raglens_http_requests",
            "HTTP requests by method, route template and status code.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "raglens_http_request_duration_seconds",
            "HTTP request latency by route template.",
            ["route"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.answers = PromCounter(
            "raglens_answers",
            "Questions handled by /ask, by retriever configuration and outcome.",
            ["retriever", "outcome"],
            registry=self.registry,
        )
        self.stage_latency = Histogram(
            "raglens_pipeline_stage_seconds",
            "Pipeline stage latency (retrieve, rerank, generate).",
            ["stage"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.index_chunks = Gauge(
            "raglens_index_chunks", "Chunks in the active index.", registry=self.registry
        )

    def render(self) -> bytes:
        """Exposition-format payload for ``GET /metrics``."""
        return generate_latest(self.registry)


class JsonFormatter(logging.Formatter):
    """One JSON object per log line, including any ``extra`` fields passed to the logger."""

    _RESERVED: ClassVar[frozenset[str]] = frozenset(vars(logging.makeLogRecord({})))

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the record with timestamp, level, logger, message and extras."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the ``raglens`` logger hierarchy (idempotent)."""
    root = logging.getLogger("raglens")
    root.setLevel(level.upper())
    handler = next((h for h in root.handlers if getattr(h, "_raglens", False)), None)
    if handler is None:
        handler = logging.StreamHandler()
        handler._raglens = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root.propagate = False


class QueryRecord(BaseModel):
    """One answered (or refused) question, as written to the query log."""

    ts: str
    request_id: str = ""
    question: str
    retriever: str
    refused: bool
    refusal_reason: str = ""
    top_document: str = ""
    citations: int = 0
    citations_supported: int = 0
    query_coverage: float = 0.0
    latency_ms: float = 0.0
    timings_ms: dict[str, float] = Field(default_factory=dict)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL,
    retriever TEXT NOT NULL,
    refused INTEGER NOT NULL,
    refusal_reason TEXT NOT NULL DEFAULT '',
    top_document TEXT NOT NULL DEFAULT '',
    citations INTEGER NOT NULL DEFAULT 0,
    citations_supported INTEGER NOT NULL DEFAULT 0,
    query_coverage REAL NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    timings_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries (ts);
"""
_COLUMNS = (
    "ts, request_id, question, retriever, refused, refusal_reason, top_document, citations, "
    "citations_supported, query_coverage, latency_ms, timings_json"
)


class QueryLog:
    """SQLite log of questions (PII-redacted), with SQL-backed analytics.

    A connection is opened per operation, so the log is safe to use from the API's worker
    threads; SQLite's write-ahead log lets readers and the writer proceed concurrently.
    """

    def __init__(self, path: str | Path, enabled: bool = True, redact: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.redact = redact
        self._initialised = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10.0)
        if not self._initialised:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            self._initialised = True
        return conn

    def append(self, record: QueryRecord) -> None:
        """Insert one record (no-op when disabled). The question is redacted first."""
        if not self.enabled:
            return
        question = redact_pii(record.question) if self.redact else record.question
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"INSERT INTO queries ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.ts,
                    record.request_id,
                    question,
                    record.retriever,
                    int(record.refused),
                    record.refusal_reason,
                    record.top_document,
                    record.citations,
                    record.citations_supported,
                    record.query_coverage,
                    record.latency_ms,
                    json.dumps(record.timings_ms),
                ),
            )

    def _row_to_record(self, row: tuple[Any, ...]) -> QueryRecord:
        return QueryRecord(
            ts=row[0],
            request_id=row[1],
            question=row[2],
            retriever=row[3],
            refused=bool(row[4]),
            refusal_reason=row[5],
            top_document=row[6],
            citations=row[7],
            citations_supported=row[8],
            query_coverage=row[9],
            latency_ms=row[10],
            timings_ms=json.loads(row[11]),
        )

    def read(self, limit: int | None = None) -> list[QueryRecord]:
        """Return the most recent ``limit`` records (all when None), oldest first."""
        if not self.path.exists():
            return []
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM queries ORDER BY id DESC LIMIT ?",
                (limit if limit else -1,),
            ).fetchall()
        return [self._row_to_record(r) for r in reversed(rows)]

    def summarize(
        self, top_n: int = 5, recent_n: int = 10, window: int = 5000
    ) -> "AnalyticsSummary":
        """Aggregate the most recent ``window`` questions with SQL."""
        if not self.path.exists():
            return AnalyticsSummary()
        recent = "WITH recent AS (SELECT * FROM queries ORDER BY id DESC LIMIT ?) "
        with closing(self._connect()) as conn:
            total, refused, citations, supported, first_ts, last_ts = conn.execute(
                recent + "SELECT COUNT(*), COALESCE(SUM(refused), 0), COALESCE(SUM(citations), 0), "
                "COALESCE(SUM(citations_supported), 0), MIN(ts), MAX(ts) FROM recent",
                (window,),
            ).fetchone()
            if not total:
                return AnalyticsSummary()
            latencies = [
                r[0] for r in conn.execute(recent + "SELECT latency_ms FROM recent", (window,))
            ]

            def counts(sql: str) -> list[CountItem]:
                rows = conn.execute(recent + sql, (window, top_n)).fetchall()
                return [CountItem(label=label, count=count) for label, count in rows]

            by_retriever = counts(
                "SELECT retriever, COUNT(*) AS n FROM recent GROUP BY retriever "
                "ORDER BY n DESC, retriever LIMIT ?"
            )
            top_documents = counts(
                "SELECT top_document, COUNT(*) AS n FROM recent WHERE top_document != '' "
                "GROUP BY top_document ORDER BY n DESC, top_document LIMIT ?"
            )
            top_questions = counts(
                "SELECT LOWER(TRIM(question)) AS q, COUNT(*) AS n FROM recent "
                "GROUP BY q ORDER BY n DESC, MAX(id) DESC LIMIT ?"
            )
            reason_rows = conn.execute(
                recent + "SELECT refusal_reason FROM recent WHERE refused = 1", (window,)
            ).fetchall()
            recent_rows = conn.execute(
                f"SELECT {_COLUMNS} FROM queries ORDER BY id DESC LIMIT ?", (recent_n,)
            ).fetchall()

        reasons = Counter(r[0].split(" (")[0].split(" <")[0] for r in reason_rows)
        return AnalyticsSummary(
            total_queries=total,
            answered=total - refused,
            refused=refused,
            refusal_rate=refused / total,
            citation_support_rate=supported / citations if citations else 0.0,
            latency_p50_ms=percentile(latencies, 50),
            latency_p95_ms=percentile(latencies, 95),
            by_retriever=by_retriever,
            top_documents=top_documents,
            top_questions=top_questions,
            refusal_reasons=[CountItem(label=k, count=v) for k, v in reasons.most_common(top_n)],
            recent=[self._row_to_record(r) for r in recent_rows],
            first_ts=first_ts,
            last_ts=last_ts,
        )


class CountItem(BaseModel):
    """A label with its count."""

    label: str
    count: int


class AnalyticsSummary(BaseModel):
    """Aggregates over the query log for the analytics dashboard."""

    total_queries: int = 0
    answered: int = 0
    refused: int = 0
    refusal_rate: float = 0.0
    citation_support_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    by_retriever: list[CountItem] = Field(default_factory=list)
    top_documents: list[CountItem] = Field(default_factory=list)
    top_questions: list[CountItem] = Field(default_factory=list)
    refusal_reasons: list[CountItem] = Field(default_factory=list)
    recent: list[QueryRecord] = Field(default_factory=list)
    first_ts: str = ""
    last_ts: str = ""
