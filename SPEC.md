# RAGLens specification

## Layout

```
src/raglens/
  ingest/      chunking/   embed/   retrieve/   rerank/
  llm/         pipeline/   eval/    api/        cli/      ui/
tests/
data/sample_corpus/
data/eval/
docs/proof/  docs/screenshots/  docs/architecture.md
scripts/
pyproject.toml  Makefile  Dockerfile  docker-compose.yml  .env.example  .pre-commit-config.yaml
.github/workflows/ci.yml
```

## Components

- **Ingestion:** `.md`, `.txt`, `.pdf` (pypdf). Each document keeps `doc_id`, source path, page
  number and character offsets.
- **Chunking:** recursive character splitter with configurable size and overlap, plus a
  sentence-aware variant. Validate that overlap < size.
- **Embeddings:** SentenceTransformers `all-MiniLM-L6-v2` on CPU, and `HashEmbedder`
  (deterministic, dependency-free, used in tests and as an offline fallback).
- **Retrieval:** BM25 (`rank_bm25`), dense (FAISS `IndexFlatIP` with normalised vectors = cosine),
  and hybrid via Reciprocal Rank Fusion (k = 60). The index persists to disk and reloads.
- **Reranking (optional):** cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, lazily loaded.
- **LLM providers** behind one interface, with timeouts, retries with exponential backoff and clear
  errors:
  - `OllamaLLM` (default `http://localhost:11434`)
  - `OpenAICompatLLM` (any `base_url` + API key, e.g. a local gateway)
  - `FakeLLM` (deterministic and extractive: builds the answer from sentences of the top chunks)
- **Pipeline:** retrieve -> (rerank) -> prompt with numbered context -> answer with inline citations
  `[1]`, `[2]`. If the best retrieval score is below a configurable threshold, answer
  "I don't know based on the provided documents."

## Evaluation harness

- **Dataset:** JSONL rows of `{question, answer, relevant_doc_ids}`. Ship a hand-written set of 40+
  questions for the sample corpus plus 5+ unanswerable questions, and a generator that creates
  synthetic QA pairs from chunks.
- **Retrieval metrics:** Hit@k, Recall@k, MRR, nDCG@k for k = 1, 3, 5, 10.
- **Answer metrics:** citation precision (share of cited chunks that are relevant), groundedness
  (token overlap between answer and cited context), refusal accuracy on unanswerable questions.
- **Latency:** p50 and p95 per stage (retrieve, rerank, generate).
- **Ablation:** retriever in {bm25, dense, hybrid, hybrid+rerank} x chunk size in {256, 512, 1024}.
  Output `docs/proof/eval_report.md` plus matplotlib charts (PNG) in `docs/proof/`.

## Sample corpus

15-25 original, self-written markdown documents (about 400-1200 words each) on ML topics:
gradient descent, overfitting and regularisation, bias-variance, decision trees and ensembles,
transformers, attention, tokenisation, embeddings, RAG, evaluation metrics, class imbalance,
anomaly detection, MLOps, and similar. No copyrighted text.

## Interfaces

- **CLI (Typer):** `raglens ingest`, `raglens ask`, `raglens eval`, `raglens serve`.
- **API (FastAPI):** `GET /health`, `POST /ingest`, `POST /ask`, `POST /eval`, with OpenAPI docs.
- **UI (Streamlit):** question box, answer with citations, retrieved chunks with scores, retriever
  selector, and an evaluation dashboard tab.

## Engineering

- pytest with coverage >= 85%, ruff (lint + format), mypy, pre-commit.
- GitHub Actions CI: lint, type-check and tests on every push and PR; coverage reported.
- Multi-stage Dockerfile and docker-compose (api + ui).
- Makefile targets: `install`, `lint`, `test`, `eval`, `serve`, `ui`, `screenshots`, `docker`.
- `scripts/capture_screenshots.py` uses Playwright (headless Chromium) to capture real screenshots
  into `docs/screenshots/`.
