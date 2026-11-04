"""Tests for metrics, JSON logging, the query log and analytics aggregation."""

import json
import logging
from pathlib import Path

import pytest

from raglens.observability import (
    AppMetrics,
    JsonFormatter,
    QueryLog,
    QueryRecord,
    configure_logging,
)


def _record(question: str, refused: bool = False, **kwargs: object) -> QueryRecord:
    values: dict[str, object] = {
        "ts": "2026-11-03T10:00:00+00:00",
        "question": question,
        "retriever": "hybrid",
        "refused": refused,
        "latency_ms": 10.0,
    }
    values.update(kwargs)
    return QueryRecord(**values)  # type: ignore[arg-type]


def test_metrics_render_prometheus_text() -> None:
    """Collectors live on a private registry and render in exposition format."""
    metrics = AppMetrics()
    metrics.http_requests.labels("POST", "/ask", "200").inc()
    metrics.answers.labels("hybrid", "refused").inc(2)
    metrics.stage_latency.labels("retrieve").observe(0.003)
    metrics.index_chunks.set(133)
    text = metrics.render().decode()
    assert 'raglens_http_requests_total{method="POST",route="/ask",status="200"} 1.0' in text
    assert 'raglens_answers_total{outcome="refused",retriever="hybrid"} 2.0' in text
    assert "_created" not in text
    assert 'raglens_pipeline_stage_seconds_count{stage="retrieve"} 1.0' in text
    assert "raglens_index_chunks 133.0" in text


def test_json_formatter_includes_extras() -> None:
    """Each line is valid JSON carrying the message and any extra fields."""
    record = logging.makeLogRecord(
        {"name": "raglens.access", "levelname": "INFO", "msg": "GET %s", "args": ("/health",)}
    )
    record.request_id = "abc123"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "GET /health"
    assert payload["request_id"] == "abc123"
    assert payload["logger"] == "raglens.access"
    assert payload["ts"].endswith("+00:00")


def test_configure_logging_is_idempotent() -> None:
    """Repeated calls reuse one handler and switch formatter."""
    configure_logging("DEBUG", json_logs=True)
    configure_logging("WARNING", json_logs=False)
    root = logging.getLogger("raglens")
    handlers = [h for h in root.handlers if getattr(h, "_raglens", False)]
    assert len(handlers) == 1
    assert not isinstance(handlers[0].formatter, JsonFormatter)
    assert root.level == logging.WARNING


def test_query_log_round_trip_and_limit(tmp_path: Path) -> None:
    """Records are stored in SQLite and read back oldest-first; limit keeps the newest."""
    log = QueryLog(tmp_path / "logs" / "q.db")
    for i in range(3):
        log.append(_record(f"question {i}", timings_ms={"retrieve": float(i)}))
    records = log.read()
    assert [r.question for r in records] == ["question 0", "question 1", "question 2"]
    assert records[2].timings_ms == {"retrieve": 2.0}
    assert [r.question for r in log.read(limit=2)] == ["question 1", "question 2"]


def test_query_log_redacts_pii_before_storing(tmp_path: Path) -> None:
    """Questions are masked on write; redaction can be switched off."""
    log = QueryLog(tmp_path / "q.db")
    log.append(_record("Why was DE89 3704 0044 0532 0130 00 charged? Mail jan@example.com"))
    assert log.read()[0].question == "Why was [IBAN] charged? Mail [EMAIL]"
    raw = QueryLog(tmp_path / "raw.db", redact=False)
    raw.append(_record("mail jan@example.com"))
    assert raw.read()[0].question == "mail jan@example.com"


def test_query_log_disabled_and_missing(tmp_path: Path) -> None:
    """A disabled log writes nothing; a missing file reads as empty."""
    log = QueryLog(tmp_path / "q.db", enabled=False)
    log.append(_record("secret question"))
    assert not log.path.exists()
    assert log.read() == []
    assert log.summarize().total_queries == 0


def test_summarize_hand_computed(tmp_path: Path) -> None:
    """SQL aggregates match values computed by hand."""
    log = QueryLog(tmp_path / "q.db")
    for record in [
        _record("What is BM25?", top_document="BM25", citations=2, citations_supported=2),
        _record("what is  bm25?", top_document="BM25", citations=1, citations_supported=0),
        _record(
            "Capital of France?",
            refused=True,
            refusal_reason="top passages cover 0% of the question's key terms (< 50%)",
            retriever="bm25",
            latency_ms=30.0,
        ),
        _record("How does RRF work?", top_document="Hybrid", latency_ms=20.0),
    ]:
        log.append(record)
    summary = log.summarize(top_n=2, recent_n=2)
    assert summary.total_queries == 4
    assert summary.answered == 3 and summary.refused == 1
    assert summary.refusal_rate == pytest.approx(0.25)
    assert summary.citation_support_rate == pytest.approx(2 / 3)
    assert summary.latency_p50_ms == pytest.approx(15.0)
    assert summary.top_documents[0].model_dump() == {"label": "BM25", "count": 2}
    assert [c.label for c in summary.by_retriever] == ["hybrid", "bm25"]
    assert summary.refusal_reasons[0].label == "top passages cover 0% of the question's key terms"
    assert [r.question for r in summary.recent] == ["How does RRF work?", "Capital of France?"]
    assert summary.first_ts == summary.last_ts == "2026-11-03T10:00:00+00:00"
    # The window limits which rows are aggregated.
    assert log.summarize(window=1).total_queries == 1


def test_summarize_groups_questions_case_insensitively(tmp_path: Path) -> None:
    """Questions differing only in case and outer whitespace are grouped."""
    log = QueryLog(tmp_path / "q.db")
    for q in ["What is BM25?", " what is bm25? ", "Other?"]:
        log.append(_record(q))
    assert log.summarize().top_questions[0].model_dump() == {"label": "what is bm25?", "count": 2}
