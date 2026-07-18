from __future__ import annotations

import tomllib

import pytest

from research_system.models import (
    Critique,
    CritiqueDisposition,
    ResearchMode,
    ResearchReport,
    ResearchRequest,
    ResearchResult,
    RunStatus,
)
from research_system.ui.components import (
    downloadable_filename,
    fallback_html_export,
    stage_label,
    stage_progress,
)


@pytest.fixture
def completed_result() -> ResearchResult:
    request = ResearchRequest(topic="How should research quality be evaluated?")
    critique = Critique(
        disposition=CritiqueDisposition.APPROVED,
        overall_score=0.9,
        citation_coverage=0.9,
        source_quality=0.9,
    )
    report = ResearchReport(
        run_id="run-export-test",
        thread_id="thread-export-test",
        topic=request.topic,
        title="A safe research export",
        executive_summary=(
            "A safe export preserves readable report content while escaping executable markup."
        ),
        markdown="# Safe research export\n\n" + "Grounded report content. " * 10,
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


def test_stage_helpers_use_human_language() -> None:
    assert stage_label("gathering") == "Gathering"
    assert stage_label("organizing") == "Organizing"
    assert stage_label("reviewing") == "Reviewing"
    assert stage_label("writing") == "Writing"
    assert stage_progress("complete") == 1.0
    assert stage_progress("unknown") == 0.0


def test_html_export_escapes_report_content(completed_result) -> None:
    unsafe_report = completed_result.model_copy(
        update={
            "report": completed_result.report.model_copy(
                update={
                    "title": "<script>alert('report')</script>",
                    "markdown": "# Safe heading\n\n<img src=x onerror=alert('report')>"
                    + " Grounded analysis." * 10,
                }
            )
        }
    )

    exported = fallback_html_export(unsafe_report)

    assert "<script>" not in exported
    assert "<img src=x" not in exported
    assert "&lt;script&gt;" in exported
    assert "&lt;img src=x" in exported


def test_downloadable_filename_is_stable_and_safe() -> None:
    assert downloadable_filename("run/with spaces", "markdown") == "research-run-with-spaces.md"
    assert downloadable_filename("", "json") == "research-report.json"


def test_streamlit_upload_limit_matches_backend_default(project_root) -> None:
    config_path = project_root / ".streamlit" / "config.toml"
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    assert config["server"]["maxUploadSize"] == 15
