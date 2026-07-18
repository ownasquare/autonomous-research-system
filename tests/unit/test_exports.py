from __future__ import annotations

from research_system.exports import export_html, export_json, export_markdown, export_result
from research_system.models import (
    Critique,
    CritiqueDisposition,
    ResearchMode,
    ResearchReport,
    ResearchRequest,
    ResearchResult,
    RunStatus,
)


def _result() -> ResearchResult:
    request = ResearchRequest(topic="How should multi-agent research systems be evaluated?")
    critique = Critique(
        disposition=CritiqueDisposition.APPROVED,
        overall_score=0.9,
        citation_coverage=1.0,
        source_quality=0.8,
    )
    report = ResearchReport(
        run_id="run-export",
        thread_id="thread-export",
        topic=request.topic,
        title="A safe <script>alert(1)</script> research report",
        executive_summary=(
            "The report demonstrates that exported content remains portable and safely encoded."
        ),
        markdown=(
            "# Safe research report\n\nA sufficiently detailed evidence-backed finding appears "
            "here [S1].\n\n## Conclusion\n\nThe export remains self-contained and safe."
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


def test_markdown_and_json_exports_are_portable() -> None:
    result = _result()

    assert export_markdown(result) == result.report.markdown
    assert '"run_id": "run-export"' in export_json(result)
    assert export_result(result, "markdown") == result.report.markdown


def test_html_export_escapes_model_content() -> None:
    html = export_html(_result())

    assert "<!doctype html>" in html.lower()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
