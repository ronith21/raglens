# RAGLens specification

## Layout

```
src/raglens/
  ingest/      chunking/   embed/   retrieve/   rerank/
  llm/         pipeline/   eval/    api/        cli/      ui/
  config.py    engine.py   textutil.py   observability.py
tests/
data/sample_corpus/
data/eval/
docs/proof/  docs/screenshots/  docs/architecture.md
scripts/
pyproject.toml  Makefile  Dockerfile  docker-compose.yml  .env.example  .pre-commit-config.yaml
.github/workflows/ci.yml
```

## Components

- **Ingestion:** `.md`, `.txt`, `.pdf` (pypdf). Each document keeps `doc_id` (hash of file name +
  content, so ids are identical on every machine), source path, page number and character offsets.
- **Chunking:** recursive character splitter with configurable size and overlap, plus a
  sentence-aware variant. Validate that overlap < size. Separators stay attached to the preceding
  piece so no text is dropped; consecutive chunks overlap by at most `chunk_overlap` characters.
  Chunks carry `title`, `source_path` and `doc_slug` (file stem) metadata.
- **Embeddings:** SentenceTransformers `all-MiniLM-L6-v2` on CPU (optional extra `[models]`), and
  `HashEmbedder` (deterministic, dependency-free, embeds stemmed content words; used in tests and
  as an offline fallback).
- **Retrieval:** BM25 (`rank_bm25`), dense (FAISS `IndexFlatIP` with normalised vectors = cosine),
  and hybrid via Reciprocal Rank Fusion (k = 60). The index persists to disk and reloads without
  re-embedding when the embedder matches.
- **Multi-query retrieval (optional):** `QueryRewriter` mines an acronym glossary from the indexed
  chunks (`Long Form (LF)` and `LF (Long Form)`) and produces up to three variants: the original
  question, its keyword form and an acronym-expanded form. `MultiQueryRetriever` retrieves each
  and fuses them with RRF.
- **Reranking (optional):** cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, lazily loaded,
  pass-through fallback.
- **LLM providers** behind one interface, with timeouts, retries with exponential backoff and clear
  errors:
  - `OllamaLLM` (default `http://localhost:11434`)
  - `OpenAICompatLLM` (any `base_url` + API key, e.g. a local gateway)
  - `FakeLLM` (deterministic and extractive: builds the answer from the top chunks' sentences
    that best match the question, weighting rare question terms higher)
- **Pipeline:** retrieve -> (rerank) -> refusal gate -> prompt with numbered context -> answer with
  inline citations `[1]`, `[2]`. When the best retrieval score is below `MIN_RELEVANCE_THRESHOLD`
  or the refusal gate rejects the evidence, answer "I don't know based on the provided documents."
  without calling the LLM. Citation styles real models emit (`[1, 2]`, `[1-3]`, `[Source 2]`) are
  normalised; numbers above 50 (years) are left alone. An uncited answer that matches a refusal
  pattern counts as a refusal. Every citation carries the best-supporting sentence of the cited
  chunk (`evidence`, chunk-relative offsets) and a `support` score (share of the claim's content
  words found there; supported if >= 0.5); weak or missing citations add `warnings`.
- **Refusal gate (`REFUSAL_GATE` = `auto` | `rule` | `learned`):**
  - `RuleGate`: refuse if the top three passages cover less than `MIN_QUERY_COVERAGE` of the
    question's stemmed content words.
  - `LearnedGate`: logistic regression over four features (coverage of the top three passages,
    best single-sentence coverage, share of specific question terms such as names, years and
    acronyms missing from the passages and their titles, and question-passage embedding
    similarity). Trained by `python -m raglens.eval.train_gate` on dev questions only; the
    threshold minimises expected cost on 5-fold cross-validated probabilities (false answer = 3,
    false refusal = 1). Stored as JSON (`GATE_MODEL_PATH`, no pickle) with the embedder it was
    trained with; `auto` uses it only when that embedder matches and falls back to the rule gate
    otherwise.

## Evaluation harness

- **Dataset:** JSONL rows of `{id, question, answer, relevant_doc_ids}` where `relevant_doc_ids`
  are document slugs (file stems). Ships 55 hand-written questions for the sample corpus plus 8
  unanswerable questions, and a generator (`raglens synth`) that creates synthetic QA pairs from
  chunks.
- **Banking benchmark:** `data/eval/banking_dev.jsonl` (tuning and gate training) and
  `data/eval/banking_test.jsonl` (held out), 32 disjoint questions each: 16 direct, 6 paraphrased,
  2 multi-document, 8 unanswerable. Every question has a `kind`; reports include a per-kind table.
- **Retrieval metrics:** Hit@k, Recall@k, MRR, nDCG@k for k = 1, 3, 5, 10 (document level).
- **Answer metrics:** citation precision (share of cited chunks that are relevant), groundedness
  (token overlap between answer and cited context), refusal accuracy on unanswerable questions,
  false-refusal rate on answerable ones, answer F1 and reference recall against the reference
  answers (stemmed content words; a wrong refusal scores 0), and citation support.
- **Latency:** p50 and p95 per stage (retrieve, rerank, generate).
- **Ablation:** retriever in {bm25, dense, hybrid, hybrid+mq, hybrid+rerank} x chunk size in
  {256, 512, 1024}.
  Output `docs/proof/eval_report.md`, `docs/proof/eval_results.json` plus matplotlib charts (PNG)
  in `docs/proof/`.

## Corpora

`data/corpora/banking/`: 14 original summaries of banking regulation and operations (Basel III
capital, liquidity, IFRS 9, AML/KYC, PSD2 and SCA, deposit protection, credit scoring and fair
lending, DORA, MiCA, SEPA/ISO 20022 and more). Educational, not legal advice. Used by
`make eval-banking`, `make demo` and the hosted demo.

`data/sample_corpus/`: 20 original, self-written markdown documents on ML topics: gradient descent, overfitting and
regularisation, bias-variance, decision trees and ensembles, transformers, attention,
tokenisation, embeddings, RAG, evaluation metrics, BM25, hybrid retrieval, reranking, class
imbalance, anomaly detection, MLOps, vector databases, chunking, hallucination, quantization.
No copyrighted text.

## Interfaces

- **CLI (Typer):** `raglens ingest`, `raglens ask`, `raglens eval [--ablation]`, `raglens synth`,
  `raglens serve`.
- **API (FastAPI):** `GET /health`, `GET /documents`, `POST /ingest` (multipart upload),
  `POST /ask`, `POST /eval`, `GET /eval/latest`, `GET /metrics` (Prometheus), `GET /analytics`,
  with OpenAPI docs at `/docs` served from bundled assets (no CDN).
- **Observability:** request-ID middleware (`X-Request-ID`), optional JSON access logs, Prometheus
  counters/histograms on a per-app registry, and a SQLite query log (`QUERY_LOG_PATH`, WAL mode)
  summarised by `/analytics` with parameterised SQL aggregates.
- **Privacy and access:** with `REDACT_PII=true` (default) e-mail addresses, IBANs, Luhn-valid card
  numbers and phone numbers are masked before a question is logged. When `API_KEY` is set,
  `/ingest`, `/eval` and `/analytics` require a matching `X-API-Key` header (constant-time compare).
- **UI (Streamlit):** question box with example prompts, answer with citation badges, retrieved
  chunks with scores, per-retriever ranks and highlighted evidence, retriever / multi-query /
  rerank / top-k / provider controls, an evaluation dashboard tab (ablation table, charts, live
  evaluation), an analytics tab and a documents tab with upload. The UI is a thin client of the API (`API_URL`); with `API_URL=embedded` it runs the API
  in-process, so a single container can serve the whole demo (Hugging Face Spaces, `make demo`).

## Engineering

- pytest with coverage >= 85% (warnings are errors), ruff (lint + format), mypy --strict,
  pre-commit.
- GitHub Actions CI: lint, type-check, tests and an evaluation smoke run on Python 3.11 and 3.12;
  coverage and eval report uploaded as artifacts; Docker image built and smoke-tested.
- Multi-stage Dockerfile (non-root, healthcheck, optional `EXTRAS=models`) and docker-compose
  (api + ui).
- Makefile targets: `install`, `install-models`, `lint`, `format`, `test`, `check`, `eval`,
  `eval-banking`, `gate`, `eval-real`, `demo`, `bench`, `ingest`, `serve`, `ui`, `screenshots`, `docker`, `compose-up`.
- `scripts/capture_screenshots.py` starts the API and UI if needed, uses Playwright (headless
  Chromium) to capture real screenshots into `docs/screenshots/` (`ui_ask.png`, `ui_citations.png`,
  `ui_eval.png`, `api_docs.png`, ...) and records a demo video (`docs/proof/raglens_demo.mp4` + `.gif`) via
  the Chrome DevTools screencast and ffmpeg.
- `scripts/dev.ps1` mirrors the Makefile for Windows PowerShell.
