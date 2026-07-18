from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_system.config import Settings
from research_system.engine import build_default_engine
from research_system.llm import extract_citation_ids
from research_system.models import IntegrityLabel, ResearchMode, ResearchRequest, RunStatus


def _cases() -> list[dict[str, object]]:
    path = Path(__file__).with_name("golden_cases.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["topic"])[:36])
def test_demo_research_quality_contract(case: dict[str, object], tmp_path: Path) -> None:
    settings = Settings(research_mode=ResearchMode.DEMO, research_data_dir=tmp_path)
    request = ResearchRequest(
        topic=str(case["topic"]),
        objective=str(case["objective"]),
    )

    with build_default_engine(settings) as engine:
        result = engine.run(request)

    assert result.status == RunStatus.COMPLETED
    assert result.report is not None
    assert result.request.topic in result.report.title
    assert len(result.sources) >= int(case["minimum_sources"])
    assert {source.integrity for source in result.sources} == {IntegrityLabel.DEMO_FIXTURE}
    assert result.report.critique.overall_score >= float(case["minimum_critic_score"])
    assert result.report.critique.citation_coverage == 1.0
    assert result.report.limitations

    known_ids = {source.id for source in result.sources}
    cited_ids = extract_citation_ids(result.report.markdown)
    assert cited_ids == set(result.report.source_ids)
    assert cited_ids <= known_ids
    assert [event.agent for event in result.trace] == [
        "researcher",
        "summarizer",
        "critic",
        "writer",
    ]
