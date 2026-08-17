"""
config/settings.py
──────────────────
Central configuration loaded once at import time.
All values come from environment variables (or a .env file).
Import the singleton:  from config.settings import settings
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # silently ignore unrecognised env vars
    )

    # ── LLM: Groq ────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")
    groq_model: str = Field(default="llama-3.1-70b-versatile", alias="GROQ_MODEL")

    # ── LLM: Ollama fallback ──────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434/v1", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="mistral", alias="OLLAMA_MODEL")

    # ── Embedding ─────────────────────────────────────────────────────────────────
    embed_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBED_MODEL")
    embed_device: str = Field(default="cpu", alias="EMBED_DEVICE")

    # ── Reranker ──────────────────────────────────────────────────────────────────
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L6-v2", alias="RERANKER_MODEL"
    )
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")

    # ── Retrieval ─────────────────────────────────────────────────────────────────
    dense_top_k: int = Field(default=20, alias="DENSE_TOP_K")
    bm25_top_k: int = Field(default=20, alias="BM25_TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")
    parent_window: int = Field(default=1, alias="PARENT_WINDOW")

    # ── Evidence gate ─────────────────────────────────────────────────────────────
    min_rerank_score: float = Field(default=0.0, alias="MIN_RERANK_SCORE")

    # ── API ───────────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # ── PDF source ────────────────────────────────────────────────────────────────
    pdf_dir: str = Field(default="./data/pdfs", alias="PDF_DIR")

    @property
    def use_groq(self) -> bool:
        """True when a Groq API key is present; fall back to Ollama otherwise."""
        return bool(self.groq_api_key)


# Module-level singleton — import this everywhere.
settings = Settings()
