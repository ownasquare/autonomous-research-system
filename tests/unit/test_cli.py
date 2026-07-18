from __future__ import annotations

from typer.testing import CliRunner

from research_system.cli import app
from research_system.models import (
    Critique,
    CritiqueDisposition,
    ResearchMode,
    ResearchReport,
    ResearchResult,
    RunStatus,
)


def _completed_result(request) -> ResearchResult:
    critique = Critique(
        disposition=CritiqueDisposition.APPROVED,
        overall_score=0.9,
        citation_coverage=1.0,
        source_quality=0.8,
    )
    report = ResearchReport(
        run_id="run-cli",
        thread_id="thread-cli",
        topic=request.topic,
        title="CLI research report",
        executive_summary="The command returns a validated source-grounded research report.",
        markdown=(
            "# CLI research report\n\nA detailed finding is available through the isolated "
            "test engine [S1].\n\n## Conclusion\n\nNo repository data was touched."
        ),
        source_ids=("S1",),
        critique=critique,
        provenance_mode=ResearchMode.DEMO,
    )
    return ResearchResult(
        run_id=report.run_id,
        thread_id=report.thread_id,
        request=request,
        status=RunStatus.COMPLETED,
        report=report,
    )


class FakeEngine:
    def __init__(self) -> None:
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args

    def run(self, request):
        self.requests.append(request)
        return _completed_result(request)

    def list_runs(self, limit=50):
        from research_system.models import ResearchRequest

        return [_completed_result(ResearchRequest(topic="Saved CLI research history"))][:limit]


def test_cli_help_lists_public_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("demo", "research", "ui", "api", "history", "graph"):
        assert command in result.output


def test_demo_uses_injected_engine_without_touching_repo_data(monkeypatch, tmp_path) -> None:
    engine = FakeEngine()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("research_system.cli.build_default_engine", lambda settings: engine)

    result = CliRunner().invoke(app, ["demo", "How should agent research be evaluated?"])

    assert result.exit_code == 0
    assert "# CLI research report" in result.output
    assert engine.requests[0].topic == "How should agent research be evaluated?"
    assert engine.requests[0].use_demo is True
    assert engine.requests[0].use_web is False
    assert engine.requests[0].use_arxiv is False
    assert engine.requests[0].use_memory is False
    assert list(tmp_path.iterdir()) == []


def test_history_uses_injected_engine_without_touching_repo_data(monkeypatch, tmp_path) -> None:
    engine = FakeEngine()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("research_system.cli.build_default_engine", lambda: engine)

    result = CliRunner().invoke(app, ["history", "--limit", "1"])

    assert result.exit_code == 0
    assert "Saved CLI research history" in result.output
    assert "run-cli" in result.output
    assert list(tmp_path.iterdir()) == []
