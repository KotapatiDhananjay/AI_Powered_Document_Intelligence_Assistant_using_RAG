"""
Application configuration loaded from environment variables.
Uses pydantic-settings for type-safe config with .env file support.
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- LLM Provider ---
    llm_provider: str = "gemini"
    google_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///./doc_intelligence.db"

    # --- Authentication ---
    jwt_secret_key: str = "change-this-to-a-random-secret-key-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 1440

    # --- RAG Settings ---
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_retrieval: int = 10
    top_k_rerank: int = 5

    # --- File Upload ---
    max_file_size_mb: int = 50
    upload_dir: str = "data/uploads"
    vector_store_dir: str = "vector_store"

    # --- Derived Properties ---
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def vector_store_path(self) -> Path:
        path = Path(self.vector_store_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def allowed_extensions(self) -> set:
        return {".pdf", ".docx", ".txt", ".pptx", ".csv", ".xlsx"}


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — loaded once, reused everywhere."""
    return Settings()
