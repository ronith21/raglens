# syntax=docker/dockerfile:1.7
# Multi-stage build: dependencies are installed into a virtualenv in the builder stage and only
# that virtualenv plus the app is copied into the slim runtime image.
#
# Real embedding / reranking models are optional:
#   docker build -t raglens .                          # offline-capable, HashEmbedder fallback
#   docker build --build-arg EXTRAS=models -t raglens . # adds sentence-transformers (+ torch)

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
ARG EXTRAS=""
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip \
 && if [ -n "$EXTRAS" ]; then pip install ".[${EXTRAS}]"; else pip install .; fi

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INDEX_DIR=/app/data/index
RUN useradd --create-home --uid 1000 raglens
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=raglens:raglens data ./data
COPY --chown=raglens:raglens docs/proof ./docs/proof
COPY --chown=raglens:raglens src/raglens/ui ./src/raglens/ui
USER raglens
EXPOSE 8000 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "raglens.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
