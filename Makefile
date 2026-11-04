.PHONY: install install-models lint format typecheck test check eval eval-banking gate eval-real bench ingest serve ui demo screenshots docker compose-up help

PYTHON ?= python3
PIP ?= pip

help:
	@echo "Available targets:"
	@echo "  install        Install in editable mode with dev + screenshot extras"
	@echo "  install-models Also install sentence-transformers (real embeddings / reranker)"
	@echo "  lint           ruff check, ruff format --check, mypy"
	@echo "  format         ruff format and autofix"
	@echo "  test           pytest with coverage (>= 85% enforced)"
	@echo "  check          lint + test (what CI runs)"
	@echo "  eval           Ablation benchmark (ML corpus) -> docs/proof/eval_report.md + charts"
	@echo "  eval-banking   Ablation on the held-out banking questions -> docs/proof/banking/"
	@echo "  gate           Train the learned refusal gate -> data/models/refusal_gate.json"
	@echo "  eval-real      Gate + banking eval with real embeddings and Ollama -> docs/proof/real/"
	@echo "  demo           Single-process UI (API embedded) on the banking corpus"
	@echo "  bench          Retrieval latency micro-benchmark -> docs/proof/retrieval_bench.txt"
	@echo "  ingest         Build the index from data/sample_corpus"
	@echo "  serve          Start the FastAPI server on :8000"
	@echo "  ui             Start the Streamlit UI on :8501"
	@echo "  screenshots    Capture UI screenshots + demo video via Playwright"
	@echo "  docker         Build the docker image"
	@echo "  compose-up     docker compose up --build (api + ui)"

install:
	$(PIP) install -e ".[dev,screenshots]"

install-models:
	$(PIP) install -e ".[dev,screenshots,models]"

lint:
	ruff check .
	ruff format --check .
	mypy src

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src

test:
	pytest --cov=raglens --cov-report=term-missing tests/

check: lint test

eval:
	$(PYTHON) -m raglens.eval.benchmark

eval-banking:
	$(PYTHON) -m raglens.eval.benchmark --corpus data/corpora/banking \
		--dataset data/eval/banking_test.jsonl --out docs/proof/banking

gate:
	$(PYTHON) -m raglens.eval.train_gate

eval-real:
	$(PYTHON) -m raglens.eval.train_gate
	$(PYTHON) -m raglens.eval.benchmark --corpus data/corpora/banking \
		--dataset data/eval/banking_test.jsonl --out docs/proof/real \
		--chunk-sizes 512 --configs bm25,dense,hybrid --llm ollama

demo:
	API_URL=embedded CORPUS_DIR=data/corpora/banking INDEX_DIR=data/index_banking \
		streamlit run src/raglens/ui/app.py

bench:
	$(PYTHON) scripts/benchmark_retrieval.py

ingest:
	raglens ingest

serve:
	uvicorn raglens.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run src/raglens/ui/app.py

screenshots:
	$(PYTHON) scripts/capture_screenshots.py

docker:
	docker build -t raglens:latest .

compose-up:
	docker compose up --build
