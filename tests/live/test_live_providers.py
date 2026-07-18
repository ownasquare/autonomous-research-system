from __future__ import annotations

import os

import pytest

from research_system.config import Settings
from research_system.engine import build_default_engine
from research_system.models import IntegrityLabel, ResearchMode, ResearchRequest, RunStatus

pytestmark = pytest.mark.live


def test_opted_in_live_research_uses_no_demo_fixtures(tmp_path) -> None:
    if os.getenv("RUN_LIVE_PROVIDER_TESTS") != "1":
        pytest.skip("set RUN_LIVE_PROVIDER_TESTS=1 to authorize provider calls")

    settings = Settings(research_mode=ResearchMode.LIVE, research_data_dir=tmp_path)
    if settings.openai_api_key is None:
        pytest.skip("OPENAI_API_KEY is required for the live model path")

    with build_default_engine(settings) as engine:
        result = engine.run(
            ResearchRequest(
                topic="Recent methods for evaluating multi-agent research systems",
                max_sources=4,
                max_revisions=0,
            )
        )

    assert result.status in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS}
    assert result.sources
    assert all(source.integrity is not IntegrityLabel.DEMO_FIXTURE for source in result.sources)
