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

    # Pipeline
    min_relevance_threshold: float = 0.01

    # Index & API
    index_dir: str = "data/index"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
