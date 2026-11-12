# syntax=docker/dockerfile:1
# Multi-stage build. EXTRAS selects optional embedders:
#   ""        -> offline LSA embeddings (smallest image)
#   "vectors" -> + pretrained GloVe word vectors (about +600 MB)
#   "minilm"  -> + sentence-transformers / cross-encoder (downloads from Hugging Face at runtime)
FROM python:3.11-slim AS builder
ARG EXTRAS=""
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels ".${EXTRAS:+[$EXTRAS]}"

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 1000 raglens
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY data ./data
COPY docs/proof ./docs/proof
RUN chown -R raglens:raglens /app
USER raglens
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "raglens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
