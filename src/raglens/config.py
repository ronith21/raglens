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
    embedder: str = "minilm"  # minilm | lsa | hash (minilm falls back to lsa offline)
    hash_embed_dim: int = 384
    lsa_components: int = 128

    # Chunking
    corpus_dir: str = "data/sample_corpus"
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
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # Pipeline: refuse when the best passage covers less than this share of the question's
    # content words. Tuned on the dev split by `raglens eval` (see docs/proof/eval_report.md).
    refusal_threshold: float = 0.35

    # Index & API
    index_dir: str = "data/index"
    eval_dataset: str = "data/eval/qa.jsonl"
    eval_output_dir: str = "docs/proof"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_mb: int = 5


settings = Settings()
