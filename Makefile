.PHONY: install lint format test eval serve ui screenshots docker help

PYTHON ?= python3
PIP ?= pip

help:
	@echo "Available targets:"
	@echo "  install      Install dependencies in editable mode"
	@echo "  lint         Run ruff and mypy"
	@echo "  format       Run ruff format and fix"
	@echo "  test         Run tests with coverage"
	@echo "  eval         Run evaluation benchmark"
	@echo "  serve        Start the FastAPI server"
	@echo "  ui           Start the Streamlit UI"
	@echo "  screenshots  Capture UI screenshots via Playwright"
	@echo "  docker       Build docker image"

install:
	$(PIP) install -e .[dev]

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
	$(PYTHON) -m raglens.eval.benchmark

serve:
	uvicorn raglens.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run src/raglens/ui/app.py

screenshots:
	$(PYTHON) scripts/capture_screenshots.py

docker:
	docker build -t raglens:latest .
