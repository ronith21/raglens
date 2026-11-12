.PHONY: install install-vectors lint format test eval demos serve ui screenshots docker help

PYTHON ?= python3
PIP ?= pip

help:
	@echo "Available targets:"
	@echo "  install          Install in editable mode with dev tools"
	@echo "  install-vectors  Also install pretrained GloVe word vectors (spaCy, ~400 MB)"
	@echo "  lint             Run ruff and mypy"
	@echo "  format           Run ruff format and fix"
	@echo "  test             Run tests with coverage"
	@echo "  eval             Run the full evaluation and write docs/proof/eval_report.md"
	@echo "  demos            Regenerate CLI/API demo transcripts in docs/proof"
	@echo "  serve            Start the FastAPI server"
	@echo "  ui               Start the Streamlit UI"
	@echo "  screenshots      Capture UI screenshots via Playwright"
	@echo "  docker           Build the docker image"

install:
	$(PIP) install -e .[dev]

install-vectors:
	$(PIP) install -e .[dev,vectors]

lint:
	ruff check .
	ruff format --check .
	mypy src

format:
	ruff format .
	ruff check --fix .

test:
	pytest --cov=raglens --cov-report=term-missing tests/

eval:
	raglens eval

demos:
	./scripts/run_demos.sh

serve:
	raglens serve

ui:
	streamlit run src/raglens/ui/app.py

screenshots:
	$(PYTHON) scripts/capture_screenshots.py

docker:
	docker build -t raglens:latest .
