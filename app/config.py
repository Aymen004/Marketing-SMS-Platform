"""Configuration helpers."""
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    catalog_path: Path = Field(default=Path("segmentationRAG/data"), env="CATALOG_PATH")
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_offres_collection: str = Field(default="offres", env="QDRANT_OFFRES_COLLECTION")
    qdrant_smartphones_collection: str = Field(default="smartphones", env="QDRANT_SMARTPHONES_COLLECTION")
    llm_base_url: str = Field(default="http://localhost:8001", env="LLM_BASE_URL")
    llm_api_key: Optional[str] = Field(default=None, env="LLM_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()