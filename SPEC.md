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
- **Embeddings:** SentenceTransformers `all-MiniLM-L6-v2` on CPU (default, optional extra
  `minilm`); `WordVectorEmbedder` (`glove`: pretrained spaCy `en_core_web_lg` vectors, IDF-weighted
  mean with the first principal component removed, optional extra `vectors`); `LSAEmbedder`
  (TF-IDF + truncated SVD fitted on the corpus, no downloads); `HashEmbedder` (deterministic
  control used in tests). The factory falls back `minilm -> glove -> lsa` and logs each step;
  it never falls back to hashing. Fitted embedder state is persisted with the index as JSON/NPZ.
- **Retrieval:** BM25 (`rank_bm25`), dense (FAISS `IndexFlatIP` with normalised vectors = cosine),
  and hybrid via Reciprocal Rank Fusion (k = 60). The index persists to disk and reloads.
- **Reranking (optional):** cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, lazily loaded.
- **LLM providers** behind one interface, with timeouts, retries with exponential backoff and clear
  errors:
  - `OllamaLLM` (default `http://localhost:11434`)
  - `OpenAICompatLLM` (any `base_url` + API key, e.g. a local gateway)
  - `FakeLLM` (deterministic and extractive: builds the answer from sentences of the top chunks)
- **Pipeline:** retrieve -> (rerank) -> evidence gate -> prompt with numbered context -> answer
  with inline citations `[1]`, `[2]`, each mapped back to its passage (out-of-range numbers are
  reported as invalid). The evidence score is the highest share of the question's stemmed
  content words found in one retrieved passage (RRF scores are rank-based and cannot express
  confidence). Below `REFUSAL_THRESHOLD` (tuned on the dev split), answer
  "I don't know based on the provided documents." without calling the LLM.

## Evaluation harness

- **Dataset:** JSONL rows of `{id, type, split, question, answer, relevant_doc_ids}` where
  `relevant_doc_ids` are file stems. Shipped: 37 lexical, 12 paraphrase and 10 unanswerable
  hand-written questions (`data/eval/build_qa.py`), alternating dev/test within each type, plus a
  generator that turns each sampled chunk's most informative sentence into a synthetic query.
- **Protocol:** retrieval metrics at document level on all answerable questions; refusal
  threshold tuned on dev (balanced accuracy); answer metrics on test only; paired bootstrap 95%
  CIs for hybrid vs BM25 and hybrid vs dense.
- **Retrieval metrics:** Hit@k, Recall@k, MRR, nDCG@k for k = 1, 3, 5, 10.
- **Answer metrics:** citation precision (share of cited chunks that are relevant), groundedness
  (token overlap between answer and cited context), refusal accuracy on unanswerable questions.
- **Latency:** p50 and p95 per stage (retrieve, rerank, generate).
- **Ablation:** retriever in {bm25, dense, hybrid, hybrid+rerank (when the model loads)} x chunk
  size in {256, 512, 1024} x embedder. Output `docs/proof/eval_report.md`, `eval_results.json`
  and matplotlib charts (PNG) in `docs/proof/`, including retriever disagreements and misses.

## Sample corpus

15-25 original, self-written markdown documents (about 400-1200 words each) on ML topics:
gradient descent, overfitting and regularisation, bias-variance, decision trees and ensembles,
transformers, attention, tokenisation, embeddings, RAG, evaluation metrics, class imbalance,
anomaly detection, MLOps, and similar. No copyrighted text.

## Interfaces

- **CLI (Typer):** `raglens ingest`, `raglens ask`, `raglens eval`, `raglens serve`.
- **API (FastAPI):** `GET /health`, `POST /ingest` (multipart .md/.txt/.pdf, size-limited, file
  names reduced to their base name), `POST /ask` (unknown fields rejected), `POST /eval`, with
  OpenAPI docs. LLM failures map to 503 without leaking provider details.
- **UI (Streamlit):** question box, answer with citations, retrieved chunks with scores, retriever
  selector, and an evaluation dashboard tab.

## Engineering

- pytest with coverage >= 85%, ruff (lint + format), mypy, pre-commit.
- GitHub Actions CI: lint, type-check and tests on every push and PR; coverage reported.
- Multi-stage Dockerfile and docker-compose (api + ui).
- Makefile targets: `install`, `lint`, `test`, `eval`, `serve`, `ui`, `screenshots`, `docker`.
- `scripts/capture_screenshots.py` uses Playwright (headless Chromium) to capture real screenshots
  into `docs/screenshots/`.
