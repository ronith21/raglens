"""Security regression tests, one or more per finding in docs/SECURITY_AUDIT.md."""

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from raglens.api.main import create_app
from raglens.api.security import (
    RateLimiter,
    clean_request_id,
    is_trusted_client,
    keys_match,
    origin_allowed,
    public_source_path,
    safe_upload_name,
    strip_control_chars,
    validate_upload_content,
)
from raglens.embed.hash import HashEmbedder
from raglens.engine import RAGEngine
from raglens.llm.fake import FakeLLM
from raglens.ui.app import highlight_passage, inert, md_text, render_answer
from tests.conftest import CORPUS_DIR, make_settings

PUBLIC_CLIENT = ("203.0.113.7", 50000)  # TEST-NET-3: never a private or loopback address
LAN_CLIENT = ("192.168.1.20", 50000)
MD_FILE = ("files", ("note.md", b"# Note\n\nThe zebracanary ratio is 7%.", "text/markdown"))


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A private copy of the sample corpus, so uploads never touch the repository."""
    target = tmp_path / "corpus"
    shutil.copytree(CORPUS_DIR, target)
    return target


def build_engine(tmp_path: Path, corpus: Path, **overrides: object) -> RAGEngine:
    engine = RAGEngine(
        make_settings(tmp_path, corpus_dir=corpus, **overrides),
        embedder=HashEmbedder(dimension=384),
        llm=FakeLLM(),
    )
    engine.load_or_build()
    return engine


@pytest.fixture
def engine_copy(tmp_path: Path, corpus: Path) -> RAGEngine:
    return build_engine(tmp_path, corpus)


@pytest.fixture
def client(engine_copy: RAGEngine) -> Iterator[TestClient]:
    with TestClient(create_app(engine=engine_copy)) as test_client:
        yield test_client


# --- 2. CSRF -----------------------------------------------------------------------------------


def test_cross_site_post_is_blocked(client: TestClient) -> None:
    """A browser POST from another site (multipart needs no CORS preflight) is rejected."""
    evil = {"Origin": "https://evil.example"}
    assert client.post("/ingest", files=[MD_FILE], headers=evil).status_code == 403
    assert client.post("/ask", json={"question": "What is BM25?"}, headers=evil).status_code == 403
    null_origin = {"Origin": "null"}  # sandboxed iframes and file:// pages
    assert client.post("/ask", json={"question": "BM25?"}, headers=null_origin).status_code == 403


def test_same_origin_and_non_browser_posts_are_allowed(client: TestClient) -> None:
    """Same-origin browser requests and server-side clients (no Origin header) still work."""
    same = {"Origin": "http://testserver"}
    assert client.post("/ask", json={"question": "What is BM25?"}, headers=same).status_code == 200
    assert client.post("/ask", json={"question": "What is BM25?"}).status_code == 200


def test_allowed_origins_setting(tmp_path: Path, corpus: Path) -> None:
    """Origins listed in ALLOWED_ORIGINS may POST cross-site."""
    engine = build_engine(tmp_path, corpus, allowed_origins=["https://app.example"])
    with TestClient(create_app(engine=engine)) as client:
        ok = client.post(
            "/ask", json={"question": "What is BM25?"}, headers={"Origin": "https://app.example/"}
        )
        assert ok.status_code == 200


def test_get_requests_ignore_origin(client: TestClient) -> None:
    """Reads are not state-changing, so cross-origin GETs are not blocked (CORS still applies)."""
    assert client.get("/health", headers={"Origin": "https://evil.example"}).status_code == 200


# --- 9. CORS -----------------------------------------------------------------------------------


def test_no_cors_headers_for_foreign_origins(client: TestClient) -> None:
    """No Access-Control-Allow-Origin is ever sent, so other sites cannot read responses."""
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers
    preflight = client.options(
        "/ask",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in preflight.headers


# --- 10/14/15. Admin authorisation and API keys --------------------------------------------------


def test_non_ascii_api_key_is_401_not_500(engine_copy: RAGEngine) -> None:
    """A non-ASCII X-API-Key header used to crash compare_digest with a 500."""
    engine_copy.settings.api_key = "s3cret"
    with TestClient(create_app(engine=engine_copy)) as client:
        response = client.get("/analytics", headers=[(b"X-API-Key", "s3cr\xe9t".encode("latin-1"))])
        assert response.status_code == 401
        assert client.get("/analytics", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_keys_match_handles_unicode_and_missing() -> None:
    assert keys_match("s3cret", "s3cret")
    assert not keys_match("s3cré", "s3cret")
    assert not keys_match(None, "s3cret")
    assert keys_match("ключ", "ключ")


def test_without_api_key_admin_is_blocked_from_public_networks(engine_copy: RAGEngine) -> None:
    """No API_KEY no longer means 'open to the internet': public clients get 403 on admin."""
    with TestClient(create_app(engine=engine_copy), client=PUBLIC_CLIENT) as client:
        assert client.get("/analytics").status_code == 403
        assert client.post("/ingest", files=[MD_FILE]).status_code == 403
        assert client.post("/eval", json={"limit": 1}).status_code == 403
        # Read-only question answering stays public.
        assert client.post("/ask", json={"question": "What is BM25?"}).status_code == 200
        assert client.get("/documents").status_code == 200


def test_without_api_key_admin_works_from_private_networks(engine_copy: RAGEngine) -> None:
    """docker compose (private bridge network) and localhost keep working without a key."""
    with TestClient(create_app(engine=engine_copy), client=LAN_CLIENT) as client:
        assert client.get("/analytics").status_code == 200


def test_api_key_lets_public_clients_in(engine_copy: RAGEngine) -> None:
    engine_copy.settings.api_key = "s3cret"
    with TestClient(create_app(engine=engine_copy), client=PUBLIC_CLIENT) as client:
        assert client.get("/analytics", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_is_trusted_client() -> None:
    nets = ["127.0.0.0/8", "10.0.0.0/8", "::1/128"]
    assert is_trusted_client("127.0.0.1", nets)
    assert is_trusted_client("10.2.3.4", nets)
    assert is_trusted_client("::1", nets)
    assert is_trusted_client("testclient", nets)
    assert not is_trusted_client("203.0.113.7", nets)
    assert not is_trusted_client("not-an-ip", nets)
    assert not is_trusted_client("", nets)
    assert not is_trusted_client(None, nets)


# --- 11. Mass assignment -----------------------------------------------------------------------


def test_unknown_fields_are_rejected(client: TestClient) -> None:
    """Clients cannot smuggle fields such as answer/refused/settings into requests."""
    body = {"question": "What is BM25?", "refused": False, "answer": "pwned", "api_key": ""}
    response = client.post("/ask", json=body)
    assert response.status_code == 422
    assert {e["loc"][-1] for e in response.json()["detail"]} == {"refused", "answer", "api_key"}
    assert client.post("/eval", json={"limit": 1, "dataset": "/etc/passwd"}).status_code == 422


def test_switching_llm_provider_is_an_admin_action(engine_copy: RAGEngine) -> None:
    """A public client cannot make the server call a paid or remote LLM."""
    with TestClient(create_app(engine=engine_copy), client=PUBLIC_CLIENT) as client:
        body = {"question": "What is BM25?", "llm_provider": "openai"}
        assert client.post("/ask", json=body).status_code == 403
        same = {"question": "What is BM25?", "llm_provider": "fake"}  # the configured provider
        assert client.post("/ask", json=same).status_code == 200


# --- 5/6. Uploads and path traversal -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../evil.md", "evil.md"),
        ("/etc/evil.md", "evil.md"),
        ("..\\..\\win.md", "win.md"),
        ("C:\\Users\\x\\doc.TXT", "doc.txt"),
        ("<img src=x onerror=alert(1)>.md", "img src_x onerror_alert_1.md"),
        ("con.md", "_con.md"),
        ("LPT1.txt", "_LPT1.txt"),
        ("..hidden.md", "hidden.md"),
        ("\uff46\uff55\uff4c\uff4c.md", "full.md"),  # fullwidth letters
        ("a" * 300 + ".md", "a" * 100 + ".md"),
    ],
)
def test_safe_upload_name(raw: str, expected: str) -> None:
    assert safe_upload_name(raw) == expected


@pytest.mark.parametrize("raw", ["", ".md", "...", "noext", "/", "..\\"])
def test_safe_upload_name_rejects_unusable(raw: str) -> None:
    with pytest.raises(ValueError):
        safe_upload_name(raw)


def test_upload_traversal_stays_inside_uploads_dir(client: TestClient, corpus: Path) -> None:
    files = [("files", ("../../../escape.md", b"# Escape\n\nzebracanary text", "text/markdown"))]
    assert client.post("/ingest", files=files).status_code == 200
    assert (corpus / "uploads" / "escape.md").exists()
    assert not (corpus.parent / "escape.md").exists()


def test_upload_rejects_disguised_and_binary_files(client: TestClient) -> None:
    """Content must match the extension: a fake PDF or binary .md is 415, not a 500 or junk."""
    fake_pdf = [("files", ("report.pdf", b"not a pdf", "application/pdf"))]
    assert client.post("/ingest", files=fake_pdf).status_code == 415
    binary = [("files", ("blob.md", bytes(range(256)), "text/markdown"))]
    assert client.post("/ingest", files=binary).status_code == 415
    empty = [("files", ("empty.md", b"", "text/markdown"))]
    assert client.post("/ingest", files=empty).status_code == 415


def test_corrupt_pdf_is_422_not_500(client: TestClient, corpus: Path) -> None:
    """A file that passes the magic-byte check but fails to parse is a client error."""
    broken = [("files", ("broken.pdf", b"%PDF-1.4\n garbage", "application/pdf"))]
    response = client.post("/ingest", files=broken)
    assert response.status_code == 422
    assert not (corpus / "uploads" / "broken.pdf").exists()


def test_upload_limits(tmp_path: Path, corpus: Path) -> None:
    engine = build_engine(tmp_path, corpus, max_upload_files=2, max_upload_mb=1)
    with TestClient(create_app(engine=engine)) as client:
        many = [("files", (f"f{i}.md", b"# T\n\ntext", "text/markdown")) for i in range(3)]
        assert client.post("/ingest", files=many).status_code == 413
        big = [("files", ("big.md", b"a" * (1024 * 1024 + 1), "text/markdown"))]
        assert client.post("/ingest", files=big).status_code == 413
        huge = client.post(
            "/ingest", content=b"x", headers={"Content-Length": str(50 * 1024 * 1024)}
        )
        assert huge.status_code == 413


def test_uploads_can_be_disabled(tmp_path: Path, corpus: Path) -> None:
    """Public demos turn uploads off so visitors cannot poison the shared corpus."""
    engine = build_engine(tmp_path, corpus, uploads_enabled=False)
    with TestClient(create_app(engine=engine)) as client:
        assert client.post("/ingest", files=[MD_FILE]).status_code == 403
        assert client.get("/health").json()["uploads_enabled"] is False


def test_validate_upload_content_accepts_bom_text_and_real_pdf_header() -> None:
    validate_upload_content("a.md", "\ufeff# Titel mit Umlaut ä".encode())
    validate_upload_content("a.pdf", b"%PDF-1.7 ...")


def test_static_files_reject_traversal(client: TestClient) -> None:
    for path in [
        "/static/swagger/../../../../etc/passwd",
        "/static/swagger/%2e%2e/%2e%2e/%2e%2e/etc/passwd",
        "/static/swagger/..%2f..%2f..%2fetc%2fpasswd",
    ]:
        response = client.get(path)
        assert response.status_code == 404
        assert "root:" not in response.text


def test_public_source_path_hides_server_layout() -> None:
    assert public_source_path("data/corpora/x.md") == "data/corpora/x.md"
    assert public_source_path("/srv/secret/layout/x.md") == "x.md"
    assert public_source_path(str(Path.cwd() / "data" / "x.md")) == "data/x.md"
    assert public_source_path("") == ""


def test_api_responses_have_no_absolute_paths(client: TestClient) -> None:
    """The test corpus lives under an absolute tmp path; responses must not reveal it."""
    docs = client.get("/documents").json()
    assert docs and all(not d["source_path"].startswith("/") for d in docs)
    answer = client.post("/ask", json={"question": "What is BM25?"}).json()
    assert all(not c["source_path"].startswith("/") for c in answer["citations"])
    assert all(
        not str(ctx["chunk"]["metadata"].get("source_path", "")).startswith("/")
        for ctx in answer["contexts"]
    )


# --- 3/4. SQL and NoSQL injection --------------------------------------------------------------


def test_sql_injection_payloads_are_stored_as_data(client: TestClient) -> None:
    payloads = ["'); DROP TABLE queries;--", "1; ATTACH DATABASE '/tmp/x.db' AS x;--"]
    for question in payloads:
        assert client.post("/ask", json={"question": question}).status_code == 200
    analytics = client.get("/analytics").json()
    assert analytics["total_queries"] == 2
    assert {r["question"] for r in analytics["recent"]} == set(payloads)
    assert client.get("/analytics", params={"limit": "1 OR 1=1"}).status_code == 422


def test_operator_objects_are_rejected(client: TestClient) -> None:
    """NoSQL-style operator objects fail type validation instead of reaching any store."""
    response = client.post("/ask", json={"question": {"$ne": None}})
    assert response.status_code == 422


# --- 8. Input validation -----------------------------------------------------------------------


def test_control_characters_are_stripped(client: TestClient) -> None:
    answer = client.post("/ask", json={"question": "What is BM25?\x00\x1b[31m\x07"}).json()
    assert answer["question"] == "What is BM25?[31m"
    assert strip_control_chars("a\tb\nc\x00\x7f") == "a\tb\nc"


def test_request_id_is_validated_before_reflection(client: TestClient) -> None:
    good = client.get("/health", headers={"X-Request-ID": "trace-42.a_b"})
    assert good.headers["X-Request-ID"] == "trace-42.a_b"
    for bad in ["A" * 8000, 'abc" injected="1', "a b", "<script>"]:
        echoed = client.get("/health", headers={"X-Request-ID": bad}).headers["X-Request-ID"]
        assert echoed != bad
        assert clean_request_id(echoed) == echoed
    assert clean_request_id(None) is None


# --- 16. Rate limits ---------------------------------------------------------------------------


def test_ask_is_rate_limited(tmp_path: Path, corpus: Path) -> None:
    engine = build_engine(tmp_path, corpus, rate_limit_per_minute=3)
    with TestClient(create_app(engine=engine)) as client:
        codes = [
            client.post("/ask", json={"question": "What is BM25?"}).status_code for _ in range(5)
        ]
        assert codes == [200, 200, 200, 429, 429]
        limited = client.post("/ask", json={"question": "What is BM25?"})
        assert int(limited.headers["Retry-After"]) >= 1


def test_api_key_guessing_is_rate_limited(tmp_path: Path, corpus: Path) -> None:
    """Admin rate limiting runs before the key check, so brute force hits 429 quickly."""
    engine = build_engine(tmp_path, corpus, api_key="s3cret", admin_rate_limit_per_minute=5)
    with TestClient(create_app(engine=engine)) as client:
        codes = [
            client.get("/analytics", headers={"X-API-Key": f"guess{i}"}).status_code
            for i in range(8)
        ]
        assert codes[:5] == [401] * 5
        assert codes[5:] == [429] * 3


def test_rate_limiter_window_and_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr("raglens.api.security.time.monotonic", lambda: now[0])
    limiter = RateLimiter(window_s=60, max_keys=4)
    assert limiter.hit("a", 2) is None
    assert limiter.hit("a", 2) is None
    assert limiter.hit("a", 2) == pytest.approx(60.0)
    now[0] += 61
    assert limiter.hit("a", 2) is None
    assert limiter.hit("a", 0) is None  # 0 disables
    for key in "bcdefg":  # exceeding max_keys evicts instead of growing without bound
        limiter.hit(key, 5)
    assert len(limiter._hits) <= 4


# --- 18. Open redirects and security headers ---------------------------------------------------


def test_no_host_header_redirects(client: TestClient) -> None:
    """Trailing-slash redirects used to echo the Host header into Location (open redirect)."""
    for path in ["/docs/", "/health/"]:
        response = client.get(path, headers={"Host": "evil.example"}, follow_redirects=False)
        assert response.status_code == 404
        assert "location" not in response.headers


def test_security_headers(client: TestClient) -> None:
    api = client.get("/health")
    assert api.headers["X-Content-Type-Options"] == "nosniff"
    assert api.headers["X-Frame-Options"] == "DENY"
    assert api.headers["Referrer-Policy"] == "no-referrer"
    assert api.headers["Content-Security-Policy"].startswith("default-src 'none'")
    docs = client.get("/docs")
    assert "script-src 'self'" in docs.headers["Content-Security-Policy"]


def test_origin_allowed_unit() -> None:
    assert origin_allowed(None, "api:8000", [])
    assert origin_allowed("http://api:8000", "api:8000", [])
    assert not origin_allowed("https://evil.example", "api:8000", [])
    assert not origin_allowed("null", "api:8000", [])
    assert not origin_allowed("https://evil.example", None, [])


# --- 1. XSS / Markdown injection in the UI -----------------------------------------------------


def test_ui_untrusted_text_cannot_escape_the_html_block() -> None:
    """A blank line used to end the HTML block, so Markdown images (tracking pixels) rendered."""
    doc = "Intro.\n\n![t](https://evil.example/t.png)\n\n<script>alert(1)</script>"
    html_out = highlight_passage(doc, [])
    assert "\n" not in html_out
    assert "<script>" not in html_out and "&lt;script&gt;" in html_out
    assert html_out.startswith('<div class="passage">')
    answer = render_answer("Line [1]\n\n![x](https://evil.example/x.png)")
    assert "\n" not in answer and '<span class="cite">1</span>' in answer
    assert inert("a\r\n\nb") == "a<br><br>b"


def test_ui_labels_escape_markdown() -> None:
    label = md_text("![x](https://evil.example/x.png) [link](javascript:alert(1))\n# h")
    assert "](" not in label.replace("\\]\\(", "")
    assert label.count("\\[") == 2 and "\n" not in label
