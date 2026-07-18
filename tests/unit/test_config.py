import stat
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from research_system.config import Settings
from research_system.models import ResearchMode


def test_settings_create_private_data_directories(tmp_path: Path) -> None:
    settings = Settings(research_data_dir=tmp_path / "data")
    settings.ensure_directories()
    assert settings.vector_path.is_dir()
    assert settings.runs_path.parent == settings.research_data_dir
    assert stat.S_IMODE(settings.research_data_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(settings.vector_path.stat().st_mode) == 0o700


def test_readiness_reports_boolean_secret_presence_only(tmp_path: Path) -> None:
    settings = Settings(
        research_data_dir=tmp_path,
        research_mode=ResearchMode.LIVE,
        openai_api_key=SecretStr("not-for-output"),
        tavily_api_key=SecretStr("not-for-output"),
    )
    readiness = settings.readiness()
    assert readiness["openai_configured"] is True
    assert readiness["tavily_configured"] is True
    assert "not-for-output" not in repr(settings)
    assert "not-for-output" not in str(readiness)


def test_blank_secret_values_are_treated_as_absent(tmp_path: Path) -> None:
    settings = Settings(research_data_dir=tmp_path, openai_api_key="", tavily_api_key=" ")
    assert settings.openai_api_key is None
    assert settings.tavily_api_key is None


def test_pdf_limits_cannot_exceed_ingestion_and_ui_contracts(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(research_data_dir=tmp_path, max_pdf_bytes=15_000_001)
    with pytest.raises(ValidationError):
        Settings(research_data_dir=tmp_path, max_pdf_characters=500_001)
