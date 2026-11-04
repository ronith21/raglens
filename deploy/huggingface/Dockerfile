# Single-container image for a free Hugging Face Space (Docker SDK).
# The Streamlit UI runs the FastAPI app in-process (API_URL=embedded), on the port Spaces expects.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    API_URL=embedded \
    CORPUS_DIR=data/corpora/banking \
    INDEX_DIR=/tmp/raglens/index \
    QUERY_LOG_PATH=/tmp/raglens/queries.db \
    USE_HASH_EMBEDDER=true \
    LLM_PROVIDER=fake

RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app
COPY --chown=user pyproject.toml README.md LICENSE ./
COPY --chown=user src ./src
RUN pip install --upgrade pip && pip install .
COPY --chown=user data ./data
COPY --chown=user docs/proof ./docs/proof
COPY --chown=user .streamlit ./.streamlit

USER user
EXPOSE 7860
CMD ["streamlit", "run", "src/raglens/ui/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
