"""RAGLens configuration settings via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    use_hash_embedder: bool = False
    hash_embed_dim: int = 384

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval & Reranker
    default_top_k: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranker: bool = False

    # LLM
    llm_provider: str = "fake"  # ollama | openai | fake
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Pipeline: refuse when the best retrieval score is below min_relevance_threshold, or when
    # the top contexts contain less than min_query_coverage of the question's content words.
    min_relevance_threshold: float = 0.01
    min_query_coverage: float = 0.5
    # auto: use the learned gate if its model file exists and matches the embedder, else rule.
    refusal_gate: str = "auto"  # auto | rule | learned
    gate_model_path: str = "data/models/refusal_gate.json"
    rerank_candidates: int = 20

    # Observability
    log_level: str = "INFO"
    log_json: bool = False
    query_log_enabled: bool = True
    query_log_path: str = "data/logs/queries.db"
    redact_pii: bool = True
    # When set, POST /ingest, POST /eval and GET /analytics require the X-API-Key header.
    api_key: str = ""

    # Index, corpus & API
    corpus_dir: str = "data/sample_corpus"
    eval_dataset: str = "data/eval/qa.jsonl"
    eval_results_path: str = "docs/proof/eval_results.json"
    index_dir: str = "data/index"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"


settings = Settings()
