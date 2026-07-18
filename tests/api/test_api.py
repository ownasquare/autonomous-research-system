from __future__ import annotations

from fastapi.testclient import TestClient

from research_system.api.app import create_app
from research_system.config import Settings
from research_system.models import (
    Critique,
    CritiqueDisposition,
    ResearchMode,
    ResearchReport,
    ResearchResult,
    RunStatus,
)


def _result(topic: str, *, failed: bool = False) -> ResearchResult:
    from research_system.models import ResearchRequest

    request = ResearchRequest(topic=topic)
    if failed:
        return ResearchResult(
            run_id="run-failed",
            thread_id="thread-api",
            request=request,
            status=RunStatus.FAILED,
            error="Research workflow failed.",
        )
    critique = Critique(
        disposition=CritiqueDisposition.APPROVED,
        overall_score=0.9,
        citation_coverage=1.0,
        source_quality=0.8,
    )
    report = ResearchReport(
        run_id="run-api",
        thread_id="thread-api",
        topic=topic,
        title="API research report",
        executive_summary="The API returns a validated and source-grounded research result.",
        markdown=(
            "# API report\n\nA sufficiently detailed evidence-backed API finding is available "
            "here [S1].\n\n## Conclusion\n\nThe validated result is portable."
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
    def __init__(self, *, fail: bool = False) -> None:
        self.settings = Settings()
        self.fail = fail
        self.records = {"run-api": _result("How should research APIs expose results?")}
        self.thread_ids = []

    def run(self, request, thread_id=None, uploads=()):
        del uploads
        self.thread_ids.append(thread_id)
        result = _result(request.topic, failed=self.fail)
        self.records[result.run_id] = result
        return result

    def get_run(self, run_id):
        return self.records.get(run_id)

    def list_runs(self, limit=50):
        return list(self.records.values())[:limit]


def test_health_and_research_routes() -> None:
    engine = FakeEngine()
    with TestClient(create_app(engine=engine)) as client:
        health = client.get("/health")
        response = client.post(
            "/research?thread_id=thread-api-follow-up",
            json={"topic": "How should research APIs expose results?"},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["readiness"]["mode"] == "demo"
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert engine.thread_ids == ["thread-api-follow-up"]


def test_run_history_detail_and_not_found() -> None:
    with TestClient(create_app(engine=FakeEngine())) as client:
        history = client.get("/runs")
        detail = client.get("/runs/run-api")
        missing = client.get("/runs/missing")

    assert history.status_code == 200
    assert detail.status_code == 200
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Research run not found."


def test_invalid_request_is_typed_and_failure_is_sanitized() -> None:
    with TestClient(create_app(engine=FakeEngine(fail=True))) as client:
        invalid = client.post("/research", json={"topic": " "})
        short_follow_up = client.post(
            "/research",
            json={"topic": "A valid research topic", "follow_up": "why"},
        )
        failed = client.post(
            "/research", json={"topic": "How should research APIs expose results?"}
        )

    assert invalid.status_code == 422
    assert short_follow_up.status_code == 422
    assert failed.status_code == 502
    assert failed.json()["detail"]["message"] == "Research workflow failed."
