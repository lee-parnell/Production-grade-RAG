from pydantic_settings import BaseSettings
from pydantic import Field, validator
from enum import Enum


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class Settings(BaseSettings):
    """
    Application configuration.
    All values from environment variables — zero hardcoded secrets.
    """

    # --- LLM ---
    openai_api_key: str = Field(..., description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model name")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # --- Embeddings ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- Qdrant ---
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "documents"

    # --- Document Processing ---
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0)
    max_upload_size_mb: int = Field(default=50, ge=1, le=200)
    supported_formats: list[str] = [".pdf", ".txt", ".md", ".docx"]

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]

    @validator("chunk_overlap")
    def overlap_less_than_chunk(cls, v, values):
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton
settings = Settings()