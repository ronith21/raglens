"""API tests with FastAPI's TestClient (no network, no real LLM)."""

import pytest
from fastapi.testclient import TestClient

from raglens.api.main import create_app
from raglens.config import Settings
from raglens.llm.base import BaseLLM, GenerationRequest, LLMError
from raglens.pipeline.rag import RAGPipeline

from .conftest import QA


@pytest.fixture()
def client(pipeline: RAGPipeline) -> TestClient:
    """Client over the shared offline pipeline."""
    settings = Settings(eval_dataset=str(QA), max_upload_mb=1)
    return TestClient(create_app(settings, pipeline=pipeline))


def test_health(client: TestClient) -> None:
    """Health reports what is loaded."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 100
    assert body["embedder"] == "lsa"
    assert body["llm"] == "fake"


def test_ask_answers_and_refuses(client: TestClient) -> None:
    """Answerable questions get citations; unanswerable ones are refused."""
    ok = client.post(
        "/ask", json={"question": "What is an inverted index?", "retriever": "bm25", "top_k": 3}
    )
    assert ok.status_code == 200
    body = ok.json()
    assert not body["refused"] and body["citations"]
    assert "11_lexical_search_bm25" in [c["source"] for c in body["contexts"]]
    no = client.post("/ask", json={"question": "What is the capital of Australia?"}).json()
    assert no["refused"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * 1001},
        {"question": "q", "top_k": 0},
        {"question": "q", "retriever": "sparse"},
        {"question": "q", "llm_provider": "openai"},  # mass assignment attempt
    ],
)
def test_ask_validation(client: TestClient, payload: dict[str, object]) -> None:
    """Bad or unexpected fields are rejected with 422."""
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_llm_failure_returns_503(pipeline: RAGPipeline) -> None:
    """A provider outage becomes a 503 without leaking internals."""

    class Down(BaseLLM):
        name = "down"

        def generate(self, request: GenerationRequest) -> str:
            raise LLMError("connection refused to http://secret-host")

    p = RAGPipeline(pipeline.retriever, Down(), refusal_threshold=0.0)
    r = TestClient(create_app(Settings(), pipeline=p)).post(
        "/ask", json={"question": "gradient descent"}
    )
    assert r.status_code == 503
    assert "secret-host" not in r.text


def test_ingest_adds_searchable_document(client: TestClient) -> None:
    """An uploaded markdown file becomes retrievable; path components are stripped."""
    doc = b"# Zorblax\n\nThe zorblax protocol rotates quantum widgets every ninety seconds."
    r = client.post("/ingest", files=[("files", ("../../evil/zorblax.md", doc, "text/markdown"))])
    assert r.status_code == 200, r.text
    assert r.json()["files"] == ["zorblax.md"]
    answer = client.post(
        "/ask", json={"question": "How often does the zorblax protocol rotate widgets?"}
    ).json()
    assert answer["contexts"][0]["source"] == "zorblax"
    assert not answer["refused"]


@pytest.mark.parametrize(
    ("name", "content", "status"),
    [
        ("malware.exe", b"MZ", 400),
        ("big.txt", b"a" * (1024 * 1024 + 10), 413),
        ("fake.pdf", b"not a pdf", 400),
        ("empty.txt", b"   ", 400),
    ],
)
def test_ingest_rejects_bad_files(
    client: TestClient, name: str, content: bytes, status: int
) -> None:
    """Unsupported types, oversized and unparseable files are refused cleanly."""
    assert client.post("/ingest", files=[("files", (name, content))]).status_code == status


def test_eval_endpoint(client: TestClient) -> None:
    """Evaluation over the bundled set returns retrieval and answer metrics."""
    r = client.post("/eval", json={"retriever": "hybrid", "split": "dev"})
    assert r.status_code == 200
    body = r.json()
    assert body["questions"] > 20
    assert 0 <= body["retrieval"]["mrr"] <= 1
    assert "refusal_accuracy_unanswerable" in body["answers"]


def test_eval_endpoint_missing_dataset(pipeline: RAGPipeline) -> None:
    """A missing dataset is a 404, not a crash."""
    app = create_app(Settings(eval_dataset="/nope.jsonl"), pipeline=pipeline)
    assert TestClient(app).post("/eval", json={}).status_code == 404
