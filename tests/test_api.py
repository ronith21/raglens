"""Integration tests for the FastAPI service using TestClient (no network, FakeLLM)."""

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from raglens.api.main import create_app
from raglens.engine import RAGEngine
from raglens.llm.base import LLMError
from tests.conftest import make_settings


@pytest.fixture
def client(engine: RAGEngine) -> Iterator[TestClient]:
    with TestClient(create_app(engine=engine)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    """Health reports index size and components."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 20
    assert body["chunks"] > 20
    assert body["embedder"] == "hash-384"
    assert body["llm_provider"] == "fake"


def test_documents(client: TestClient) -> None:
    """All corpus documents are listed with chunk counts."""
    docs = client.get("/documents").json()
    assert len(docs) == 20
    assert all(d["num_chunks"] >= 1 for d in docs)


def test_ask_answers_with_citations(client: TestClient) -> None:
    """A corpus question returns a cited, non-refused answer with contexts and timings."""
    response = client.post("/ask", json={"question": "How do isolation forests detect anomalies?"})
    assert response.status_code == 200
    body = response.json()
    assert not body["refused"]
    assert body["citations"]
    assert body["citations"][0]["source_path"].endswith("15_anomaly_detection.md")
    assert len(body["contexts"]) == 5
    assert {"retrieve", "generate"} <= set(body["timings_ms"])


def test_ask_refuses_out_of_scope(client: TestClient) -> None:
    """Out-of-corpus questions are refused."""
    body = client.post("/ask", json={"question": "Who won the 2018 FIFA World Cup?"}).json()
    assert body["refused"]
    assert body["answer"] == "I don't know based on the provided documents."


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "ok", "retriever": "tfidf"},
        {"question": "ok", "top_k": 0},
        {"question": "ok", "top_k": 99},
        {"question": "x" * 2001},
        {"question": "ok", "llm_provider": "gpt-magic"},
    ],
)
def test_ask_validation_errors(client: TestClient, payload: dict[str, object]) -> None:
    """Malformed bodies get 422 from schema validation."""
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_blank_question_rejected(client: TestClient) -> None:
    """Whitespace-only questions are rejected explicitly."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422
    assert "blank" in response.json()["detail"]


def test_ask_llm_failure_maps_to_502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider failures surface as 502 without leaking provider details to the client."""

    def boom(*args: object, **kwargs: object) -> None:
        raise LLMError("Ollama request failed after 3 attempt(s)")

    monkeypatch.setattr(RAGEngine, "ask", boom)
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "unavailable" in detail
    assert response.headers["X-Request-ID"] in detail
    assert "Ollama" not in detail


def test_ask_empty_index_conflict(tmp_path: Path) -> None:
    """Asking before anything is indexed returns 409."""
    settings = make_settings(tmp_path, corpus_dir=tmp_path / "missing")
    empty = RAGEngine(settings)
    with TestClient(create_app(engine=empty)) as client:
        assert client.post("/ask", json={"question": "hello?"}).status_code == 409


def test_ingest_upload_and_query(client: TestClient, engine: RAGEngine, tmp_path: Path) -> None:
    """Uploaded markdown becomes searchable; unsupported types get 415."""
    engine.settings.corpus_dir = str(tmp_path / "corpus")
    files = [
        (
            "files",
            ("okapi.md", b"# Okapi\n\nThe okapi is a giraffid from the Congo.", "text/markdown"),
        )
    ]
    response = client.post("/ingest", files=files)
    assert response.status_code == 200
    assert response.json()["documents_added"] == 1
    assert (tmp_path / "corpus" / "uploads" / "okapi.md").exists()

    body = client.post(
        "/ask", json={"question": "Where is the okapi from?", "retriever": "bm25"}
    ).json()
    assert "Congo" in body["answer"]

    bad = client.post("/ingest", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 415


def test_eval_endpoint(client: TestClient) -> None:
    """POST /eval runs the dataset (limited) and returns metrics."""
    response = client.post(
        "/eval", json={"retriever": "bm25", "limit": 5, "include_questions": False}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["num_questions"] == 5
    assert body["questions"] == []
    assert 0.0 <= body["retrieval"]["mrr"] <= 1.0


def test_eval_missing_dataset(client: TestClient, engine: RAGEngine, tmp_path: Path) -> None:
    """A missing dataset is a 404."""
    engine.settings.eval_dataset = str(tmp_path / "none.jsonl")
    assert client.post("/eval", json={}).status_code == 404


def test_eval_latest(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """/eval/latest serves docs/proof/eval_results.json or 404s."""
    monkeypatch.chdir(tmp_path)  # the default results path is relative to the working directory
    assert client.get("/eval/latest").status_code == 404
    (tmp_path / "docs" / "proof").mkdir(parents=True)
    (tmp_path / "docs" / "proof" / "eval_results.json").write_text('{"meta": {}, "reports": []}')
    assert client.get("/eval/latest").json() == {"meta": {}, "reports": []}


def test_swagger_docs_served_without_cdn(client: TestClient) -> None:
    """/docs references only locally bundled Swagger assets, which are served."""
    html = client.get("/docs").text
    assert "/static/swagger/swagger-ui-bundle.js" in html
    assert "cdn.jsdelivr.net" not in html
    assert client.get("/static/swagger/swagger-ui-bundle.js").status_code == 200


def test_openapi_schema_lists_endpoints(client: TestClient) -> None:
    """OpenAPI docs are generated for every endpoint."""
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/ingest", "/ask", "/eval", "/documents"} <= set(paths)
    assert isinstance(httpx.URL("http://testserver/docs"), httpx.URL)


def test_request_id_header_round_trip(client: TestClient) -> None:
    """A caller's X-Request-ID is echoed; otherwise one is generated."""
    assert (
        client.get("/health", headers={"X-Request-ID": "trace-42"}).headers["x-request-id"]
        == "trace-42"
    )
    assert len(client.get("/health").headers["x-request-id"]) == 16


def test_metrics_and_analytics_after_asks(client: TestClient) -> None:
    """/ask updates Prometheus counters and the query log behind /analytics."""
    client.post("/ask", json={"question": "How do isolation forests detect anomalies?"})
    client.post("/ask", json={"question": "Who won the 2018 FIFA World Cup?"})
    client.post("/ask", json={"question": "How do isolation forests detect anomalies?"})

    metrics = client.get("/metrics")
    assert metrics.headers["content-type"].startswith("text/plain")
    text = metrics.text
    assert 'raglens_answers_total{outcome="answered",retriever="hybrid"} 2.0' in text
    assert 'raglens_answers_total{outcome="refused",retriever="hybrid"} 1.0' in text
    assert 'raglens_http_requests_total{method="POST",route="/ask",status="200"} 3.0' in text
    assert 'raglens_pipeline_stage_seconds_count{stage="retrieve"} 3.0' in text

    summary = client.get("/analytics").json()
    assert summary["total_queries"] == 3
    assert summary["refused"] == 1
    assert summary["refusal_rate"] == pytest.approx(1 / 3)
    assert summary["top_questions"][0] == {
        "label": "how do isolation forests detect anomalies?",
        "count": 2,
    }
    assert (
        summary["top_documents"][0]["label"] == "Anomaly Detection Methodologies and Applications"
    )
    assert summary["recent"][0]["question"] == "How do isolation forests detect anomalies?"
    assert summary["recent"][0]["request_id"]
    assert client.get("/analytics", params={"limit": 0}).status_code == 422


def test_ask_multi_query_and_evidence(client: TestClient) -> None:
    """multi_query=true reports the variants; citations include evidence spans."""
    body = client.post(
        "/ask", json={"question": "What does DVC track?", "multi_query": True}
    ).json()
    assert body["retriever"] == "hybrid+mq"
    assert "data version control" in " ".join(body["contexts"][0]["metadata"]["query_variants"])
    citation = body["citations"][0]
    assert citation["evidence"]
    assert 0.0 <= citation["support"] <= 1.0
    assert citation["evidence_end"] > citation["evidence_start"] >= 0


def test_llm_error_counted(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider failures increment the error outcome."""

    def boom(*args: object, **kwargs: object) -> None:
        raise LLMError("down")

    monkeypatch.setattr(RAGEngine, "ask", boom)
    client.post("/ask", json={"question": "anything"})
    assert (
        'raglens_answers_total{outcome="error",retriever="hybrid"} 1.0'
        in client.get("/metrics").text
    )


def test_api_key_protects_write_and_analytics_endpoints(engine: RAGEngine) -> None:
    """With API_KEY set, /ingest, /eval and /analytics need X-API-Key; /ask stays open."""
    engine.settings.api_key = "s3cret"
    with TestClient(create_app(engine=engine)) as client:
        assert client.get("/analytics").status_code == 401
        assert client.get("/analytics", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/analytics", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert client.post("/eval", json={"limit": 1}).status_code == 401
        files = [("files", ("a.md", b"# A\n\ntext", "text/markdown"))]
        assert client.post("/ingest", files=files).status_code == 401
        assert client.post("/ask", json={"question": "What is BM25?"}).status_code == 200


def test_query_log_stores_redacted_question(client: TestClient, engine: RAGEngine) -> None:
    """Personal data in a question never reaches the query log."""
    client.post("/ask", json={"question": "Why was my card 4111 1111 1111 1111 declined?"})
    recent = client.get("/analytics").json()["recent"][0]
    assert recent["question"] == "Why was my card [CARD] declined?"
