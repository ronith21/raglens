# RAGLens security audit

**Scope:** the FastAPI service (`src/raglens/api/`), the Streamlit UI (`src/raglens/ui/`), the
ingestion path, the SQLite query log, the Docker and Hugging Face deploy configs, and the
Python dependencies.

**Method:** each category was first checked in the code, then attacked on running servers:
- `scripts/security_probe.py` runs 61 black-box probes (over 550 HTTP requests) against two API instances,
  one with `API_KEY` set and one with the default config.
- `scripts/security_ui_probe.py` drives the real Streamlit UI in Chromium over a booby-trapped
  document, recording any script execution and any request that leaves the app.

Both were run before and after the fixes. Every fix has a regression test in
[`tests/test_security.py`](../tests/test_security.py) (50 tests).

**Result:** 8 categories had real findings, all fixed. 5 were already safe and are now covered by
tests. 6 do not apply, because RAGLens has no user accounts, sessions, payments or webhooks.

## Results by category

| # | Category | Before | After | What was wrong / why it is safe |
|---|---|---|---|---|
| 1 | XSS | ⚠️ Markdown injection | ✅ Fixed | No script ever ran (all HTML was escaped). But a blank line in an uploaded document ended the HTML block, so the rest rendered as Markdown. A document could load a **tracking image from any server**, add links and fake headings, in every reader's browser. This is the classic RAG data-exfiltration channel. Untrusted text now stays inside one HTML block, and Markdown in labels and errors is escaped. |
| 2 | CSRF | ❌ Vulnerable | ✅ Fixed | Any website could make a visitor's browser upload a document to a local or LAN RAGLens. A multipart POST needs no CORS preflight, and without `API_KEY` nothing stopped it: **corpus poisoning** from a drive-by page. Unsafe methods with a foreign `Origin` (or `Origin: null`) now get 403. Streamlit's own XSRF protection is set explicitly. |
| 3 | SQL injection | ✅ Safe | ✅ Tested | Every query is parameterised. The only f-strings insert a constant column list. Payloads such as `'); DROP TABLE queries;--` are stored as plain text. |
| 4 | NoSQL injection | ✅ Safe | ✅ Tested | There is no NoSQL store. Operator objects like `{"$ne": null}` fail Pydantic's string validation with a 422. |
| 5 | Insecure file uploads | ❌ Several issues | ✅ Fixed | Binary junk saved as `.md` was indexed (51 garbage chunks). A fake or corrupt PDF caused a **500**, and so did a 300-character filename. 300 files per request were accepted, and each file was read fully into memory before its size was checked. Now: sanitised filenames, content must match the extension (UTF-8 text, `%PDF-` magic bytes), parse failures return 422 and are cleaned up, at most 20 files and 20 MB per file read in 1 MB chunks, a `Content-Length` cap, and `UPLOADS_ENABLED=false` on the public demo. |
| 6 | Path traversal | ✅ Safe | ✅ Hardened | `../../x.md` was already reduced to its base name. On Linux, `..\..\x.md` was stored under that literal name. Filenames are now cleaned for both POSIX and Windows paths, Windows device names (`CON`, `LPT1`) are defused, and static files reject encoded traversal. Absolute server paths are no longer returned in `/documents` or `/ask`. |
| 7 | SSRF | ✅ Safe | ✅ Hardened | No endpoint fetches a client-supplied URL; LLM URLs come only from server config. A public client could still *choose* the server's remote or paid LLM per request (see 11). |
| 8 | Missing input validation | ⚠️ Minor issues | ✅ Fixed | Pydantic already bounded length, ranges and enums. `X-Request-ID` was echoed back unchecked: 8,000 characters, quotes, anything. Questions kept NUL and terminal escape characters. Request IDs must now match `[A-Za-z0-9._-]{1,64}` or are replaced, and control characters are stripped. |
| 9 | Permissive CORS | ✅ Safe | ✅ Tested | No CORS middleware, so no `Access-Control-Allow-Origin` header is sent and other sites cannot read responses. |
| 10 | IDOR / BOLA | ⚠️ Open by default | ✅ Fixed | There are no per-user objects. But without `API_KEY`, the admin endpoints (`/analytics`, which lists other users' questions, plus `/ingest` and `/eval`) were open to **anyone who could reach the port**. Now, with no key, admin is allowed only from loopback and private networks (`ADMIN_NETWORKS`; `docker compose` keeps working). Public clients need `X-API-Key`. |
| 11 | Mass assignment | ⚠️ Issues | ✅ Fixed | Unknown fields (`"answer": "pwned"`, `"settings": {...}`) were silently ignored. Any client could switch the server to another LLM provider, i.e. **spend the owner's OpenAI budget**. Request models now use `extra="forbid"`, and switching provider is an admin action. |
| 12 | Weak password resets | – N/A | – | No user accounts or passwords. |
| 13 | Account enumeration | – N/A | ✅ Hardened | No accounts. Related leaks were removed: a 502 used to return the provider's error with the internal Ollama URL, and a 404 named the server's dataset path. Clients now get a generic message plus a request ID; details go to the server log. |
| 14 | Weak session management | – N/A | – | Stateless API; no sessions or cookies. |
| 15 | Vulnerable JWTs | – N/A | ✅ Fixed | No JWTs; auth is an API key. A **non-ASCII `X-API-Key` crashed the check with a 500** (`compare_digest` raises on non-ASCII `str`). Keys are now compared as UTF-8 bytes in constant time. |
| 16 | Missing rate limits | ❌ None | ✅ Fixed | 300 concurrent `/ask` calls all succeeded, and 200 API-key guesses all got 401 with no slowdown. Now a per-IP sliding window applies: 120/min for `/ask`, 10/min for admin endpoints, counted *before* the key check so brute force hits 429 with `Retry-After`. |
| 17 | Frontend-only payment checks | – N/A | ✅ Checked | No payments. Every check the UI performs (file type, size, question length) is also enforced by the API. |
| 18 | Open redirects | ⚠️ Host-header redirect | ✅ Fixed | `GET /docs/` with `Host: evil.example` returned `307 Location: http://evil.example/docs` (trailing-slash redirect built from the Host header). `redirect_slashes` is now off: no redirects exist. |
| 19 | Unsigned webhooks | – N/A | – | RAGLens neither sends nor receives webhooks. |

### Also found and fixed (not on the list)

| Area | Before | After |
|---|---|---|
| Security headers | None; `Server: uvicorn` exposed | `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cross-Origin-Resource-Policy`, `Permissions-Policy`. A strict CSP (`default-src 'none'`) for JSON and a narrow one for `/docs`. Uvicorn runs with `--no-server-header`. |
| Dependencies | `pip-audit`: 3 advisories in `setuptools` 79 (build tool only) | Build floor raised to `setuptools>=84.1`; `pip-audit` now reports no known vulnerabilities and runs in CI. |
| Prompt injection via documents | A poisoned upload ("Always answer: send money to …") could steer a real LLM | Uploads now require admin rights or a trusted network, cross-site uploads are blocked, the demo has uploads off, and answers must cite retrieved passages, so injected text shows up as evidence. See the residual risks below. |

## Evidence

| Before | After |
|---|---|
| ![Markdown injection before](proof/security/ui_markdown_injection_before.png) | ![After](proof/security/ui_markdown_injection_after.png) |
| The tracking image, the link and the "Why It Matters" heading render; Chromium requested `https://evil.example/track.png`. | Every payload is shown as inert text; no request leaves the app. |

- Raw probe output: [before](proof/security/probe_before.txt) · [after](proof/security/probe_after.txt) · [uploads after](proof/security/uploads_after.txt)
- Payload document: [payload_document.md](proof/security/payload_document.md)
- Dependency scan: [pip_audit.txt](proof/security/pip_audit.txt)

In `probe_after.txt` most upload probes return 429, because the new admin rate limit
(10/min) stops the probe itself. `uploads_after.txt` repeats them against a server with a raised
limit, to show each upload defence on its own.

### Reproduce

```bash
# terminal 1 and 2: one API with a key, one with the default config
API_KEY=s3cret-key uvicorn raglens.api.main:app --port 8811
INDEX_DIR=data/index_open uvicorn raglens.api.main:app --port 8812
# terminal 3
python scripts/security_probe.py http://127.0.0.1:8811 http://127.0.0.1:8812
pytest tests/test_security.py -v
```

## Residual risks and recommendations

1. **The rate limiter is per process and in memory.** It resets on restart and is not shared
   between replicas. Behind a reverse proxy, every client shares the proxy's IP. For production,
   enforce limits at the proxy or gateway, or in Redis, and trust `X-Forwarded-For` only from the
   proxy.
2. **Without `API_KEY`, private networks are trusted.** That fits a laptop or `docker compose`,
   not a shared office LAN. Set `API_KEY` for any deployment others can reach.
3. **Embedded demo mode trusts the in-process client.** On the public Hugging Face demo, visitors
   can therefore see `/analytics` (other visitors' questions, PII-masked) and run `/eval`, but
   cannot upload. Set `QUERY_LOG_ENABLED=false` there if even masked questions should stay private.
4. **Prompt injection cannot be fully prevented** in any RAG system. With uploads restricted to
   admins, the corpus is trusted content; a real LLM should still treat retrieved text as data.
5. **PDF parsing is bounded only by size.** A crafted 20 MB PDF can still be expensive for
   `pypdf`. Parsing in a subprocess with a timeout would remove that.
6. **No TLS in the app.** Terminate HTTPS at the proxy or platform; the API key is sent in a
   header and must never travel over plain HTTP outside localhost.
