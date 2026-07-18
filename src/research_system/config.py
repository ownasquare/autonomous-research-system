"""Environment-backed configuration with explicit provider readiness."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from research_system.models import ResearchMode


class Settings(BaseSettings):
    """Validated runtime settings. Secret values are never included in diagnostics."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    research_mode: ResearchMode = ResearchMode.DEMO
    research_data_dir: Path = Path(".research_data")
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"
    tavily_api_key: SecretStr | None = None
    request_timeout_seconds: float = Field(default=30.0, ge=3.0, le=120.0)
    max_search_results: int = Field(default=8, ge=2, le=20)
    max_pdf_bytes: int = Field(default=15_000_000, ge=100_000, le=15_000_000)
    max_pdf_uploads: int = Field(default=5, ge=1, le=20)
    max_upload_bytes_total: int = Field(default=30_000_000, ge=100_000, le=60_000_000)
    max_pdf_pages: int = Field(default=100, ge=1, le=500)
    max_pdf_characters: int = Field(default=500_000, ge=1_000, le=500_000)
    arxiv_min_interval_seconds: float = Field(default=3.0, ge=3.0, le=30.0)

    @field_validator("openai_api_key", "tavily_api_key", mode="before")
    @classmethod
    def blank_secret_is_none(cls, value: object) -> object:
        """Treat blank values from `.env.example` as absent credentials."""

        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def checkpoint_path(self) -> Path:
        return self.research_data_dir / "checkpoints.sqlite3"

    @property
    def runs_path(self) -> Path:
        return self.research_data_dir / "runs.sqlite3"

    @property
    def vector_path(self) -> Path:
        return self.research_data_dir / "vector-memory"

    def ensure_directories(self) -> None:
        self.research_data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.research_data_dir.chmod(0o700)
        self.vector_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.vector_path.chmod(0o700)

    def readiness(self) -> dict[str, Any]:
        """Return key-name-only readiness data suitable for UI and logs."""

        return {
            "mode": self.research_mode.value,
            "openai_configured": self.openai_api_key is not None,
            "tavily_configured": self.tavily_api_key is not None,
            "arxiv_available": True,
            "pdf_parser_available": True,
            "data_directory": str(self.research_data_dir),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
