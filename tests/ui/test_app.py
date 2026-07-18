from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from research_system.models import (
    AgentTraceEvent,
    ConversationTurn,
    Critique,
    CritiqueDisposition,
    IntegrityLabel,
    ResearchMode,
    ResearchReport,
    ResearchRequest,
    ResearchResult,
    RunStatus,
    Source,
    SourceKind,
    WorkflowEvent,
)
from research_system.ui import app as ui_app


def _result_for(request: ResearchRequest, *, run_id: str = "run-ui-001") -> ResearchResult:
    source = Source(
        id="S1",
        kind=SourceKind.DEMO,
        title="A practical guide to evaluated multi-agent research",
        url="demo://research/evaluation",
        snippet=(
            "Evaluation should combine source quality, citation coverage, and revision evidence."
        ),
        content=(
            "Evaluation should combine source quality, citation coverage, and revision evidence."
        ),
        provider="bundled demo library",
        integrity=IntegrityLabel.DEMO_FIXTURE,
        checksum="1234567890abcdef",
    )
    critique = Critique(
        disposition=CritiqueDisposition.APPROVED,
        overall_score=0.91,
        citation_coverage=0.96,
        source_quality=0.87,
        strengths=("Claims are connected to the source library.",),
        gaps=("Live provider breadth was not evaluated in demo mode.",),
    )
    report = ResearchReport(
        run_id=run_id,
        thread_id="thread-ui-001",
        topic=request.topic,
        title="Evaluating multi-agent research systems",
        executive_summary=(
            "Strong evaluation combines evidence quality, citation coverage, transparent critique, "
            "and repeatable workflow traces."
        ),
        markdown=(
            "# Evaluating multi-agent research systems\n\n"
            "A useful evaluation program measures source quality and citation coverage [S1].\n\n"
            "## Recommendation\n\nUse repeatable cases, a visible critique pass, "
            "and explicit limitations. "
            "This keeps the report decision-ready while preserving provenance [S1]."
        ),
        source_ids=("S1",),
        critique=critique,
        limitations=("This deterministic run uses the bundled demo source library.",),
        provenance_mode=ResearchMode.DEMO,
    )
    trace = tuple(
        AgentTraceEvent(
            sequence=index,
            agent=agent,
            status="completed",
            message=message,
        )
        for index, (agent, message) in enumerate(
            (
                ("researcher", "Collected and ranked one source."),
                ("summarizer", "Organized the evidence into findings."),
                ("critic", "Reviewed claims and citation coverage."),
                ("writer", "Prepared the final report."),
            ),
            start=1,
        )
    )
    return ResearchResult(
        run_id=run_id,
        thread_id="thread-ui-001",
        request=request,
        status=RunStatus.COMPLETED,
        sources=(source,),
        report=report,
        trace=trace,
        conversation=(ConversationTurn(role="assistant", content=report.executive_summary),),
    )


class FakeEngine:
    def __init__(
        self,
        *,
        history: Sequence[ResearchResult] = (),
        mode: ResearchMode = ResearchMode.DEMO,
    ) -> None:
        self.history = list(history)
        self.requests: list[ResearchRequest] = []
        self.uploads: list[tuple[str, bytes]] = []
        self.settings = SimpleNamespace(
            research_mode=mode,
            max_pdf_bytes=15_000_000,
        )

    def iter_run(
        self,
        request: ResearchRequest,
        thread_id: str | None = None,
        uploads: Sequence[tuple[str, bytes]] = (),
    ) -> Iterator[WorkflowEvent]:
        del thread_id
        self.requests.append(request)
        self.uploads = list(uploads)
        result = _result_for(request)
        stages = (
            ("gathering", "researcher", "Gathering relevant sources", 0.2),
            ("organizing", "summarizer", "Organizing the strongest evidence", 0.5),
            ("reviewing", "critic", "Reviewing claims and citations", 0.75),
            ("writing", "writer", "Writing the research report", 0.9),
        )
        for stage, agent, message, progress in stages:
            yield WorkflowEvent(
                run_id=result.run_id,
                stage=stage,
                agent=agent,
                message=message,
                progress=progress,
            )
        self.history.insert(0, result)
        yield WorkflowEvent(
            run_id=result.run_id,
            stage="complete",
            agent="supervisor",
            message="Research complete",
            progress=1.0,
            result=result,
        )

    def run(
        self,
        request: ResearchRequest,
        thread_id: str | None = None,
        uploads: Sequence[tuple[str, bytes]] = (),
    ) -> ResearchResult:
        terminal = list(self.iter_run(request, thread_id=thread_id, uploads=uploads))[-1]
        assert terminal.result is not None
        return terminal.result

    def get_run(self, run_id: str) -> ResearchResult | None:
        return next((item for item in self.history if item.run_id == run_id), None)

    def list_runs(self, limit: int = 20) -> list[ResearchResult]:
        return self.history[:limit]

    def close(self) -> None:
        return None


class FailingEngine(FakeEngine):
    def iter_run(
        self,
        request: ResearchRequest,
        thread_id: str | None = None,
        uploads: Sequence[tuple[str, bytes]] = (),
    ) -> Iterator[WorkflowEvent]:
        del request, thread_id, uploads
        raise RuntimeError("provider-secret-must-not-reach-the-interface")
        yield  # pragma: no cover


def test_default_engine_is_process_owned_and_closed_once(monkeypatch) -> None:
    class TrackingEngine(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    engine = TrackingEngine()
    creations = 0

    def factory() -> TrackingEngine:
        nonlocal creations
        creations += 1
        return engine

    ui_app._close_process_engine()
    monkeypatch.setattr(ui_app, "_PROCESS_ENGINE_ERROR", None)
    monkeypatch.setattr(ui_app, "_default_engine_factory", factory)
    with ThreadPoolExecutor(max_workers=8) as executor:
        resolved = list(executor.map(lambda _index: ui_app._get_process_engine(), range(16)))

    assert creations == 1
    assert all(item == (engine, None) for item in resolved)
    ui_app._close_process_engine()
    ui_app._close_process_engine()
    assert engine.close_calls == 1


@pytest.fixture(autouse=True)
def _reset_engine_override() -> Iterator[None]:
    ui_app.clear_engine_factory_for_testing()
    yield
    ui_app.clear_engine_factory_for_testing()


@pytest.fixture
def app_path(project_root: Path) -> Path:
    return project_root / "src" / "research_system" / "streamlit_app.py"


def _app(app_path: Path, engine: FakeEngine) -> AppTest:
    ui_app.set_engine_factory_for_testing(lambda: engine)
    return AppTest.from_file(str(app_path), default_timeout=10)


@pytest.mark.parametrize(
    ("depth", "expected_budget"),
    (("Quick", (6, 0)), ("Standard", (12, 1)), ("Deep", (20, 2))),
)
def test_request_inputs_use_canonical_depth_budgets(
    depth: str, expected_budget: tuple[int, int]
) -> None:
    request = ui_app._request_from_inputs(
        topic="How should multi-agent research systems be evaluated?",
        objective="Produce a decision-ready report.",
        audience="Research leaders",
        depth=depth,
        sources=("Web",),
    )

    assert (request.max_sources, request.max_revisions) == expected_budget


@pytest.mark.ui
def test_workspace_renders_primary_research_controls(app_path: Path) -> None:
    app = _app(app_path, FakeEngine()).run()

    assert app.title[0].value == "Research Desk"
    assert app.text_area(key="research_topic")
    assert app.text_input(key="research_objective")
    assert app.text_input(key="research_audience")
    assert app.selectbox(key="research_depth")
    assert app.multiselect(key="research_sources")
    assert app.button(key="start_research")
    assert any("Demo mode - uses bundled fixture evidence" in item.value for item in app.info)
    assert not app.exception


@pytest.mark.ui
def test_source_choices_match_demo_and_live_provider_boundaries(app_path: Path) -> None:
    demo = _app(app_path, FakeEngine(mode=ResearchMode.DEMO)).run()
    demo_sources = demo.multiselect(key="research_sources")

    assert demo_sources.options == ["Bundled demo", "Research memory"]
    assert demo_sources.value == ["Bundled demo"]

    live = _app(app_path, FakeEngine(mode=ResearchMode.LIVE)).run()
    live_sources = live.multiselect(key="research_sources")

    assert live_sources.options == ["Web", "arXiv", "Research memory"]
    assert live_sources.value == ["Web", "arXiv", "Research memory"]


def test_demo_source_selection_maps_to_truthful_request_flags() -> None:
    bundled_only = ui_app._request_from_inputs(
        topic="How should multi-agent research systems be evaluated?",
        objective="Produce a decision-ready report.",
        audience="Research leaders",
        depth="Standard",
        sources=("Bundled demo",),
    )
    with_memory = ui_app._request_from_inputs(
        topic=bundled_only.topic,
        objective=bundled_only.objective,
        audience=bundled_only.audience,
        depth="Standard",
        sources=("Bundled demo", "Research memory"),
    )
    memory_only = ui_app._request_from_inputs(
        topic=bundled_only.topic,
        objective=bundled_only.objective,
        audience=bundled_only.audience,
        depth="Standard",
        sources=("Research memory",),
    )

    assert (
        bundled_only.use_web,
        bundled_only.use_arxiv,
        bundled_only.use_memory,
        bundled_only.use_demo,
    ) == (
        False,
        False,
        False,
        True,
    )
    assert with_memory.use_memory is True
    assert with_memory.use_demo is True
    assert memory_only.use_memory is True
    assert memory_only.use_demo is False


@pytest.mark.ui
def test_empty_topic_shows_validation_without_calling_engine(app_path: Path) -> None:
    engine = FakeEngine()
    app = _app(app_path, engine).run()

    app.button(key="start_research").click().run()

    assert not engine.requests
    assert any("topic" in item.value.lower() for item in app.error)


@pytest.mark.ui
def test_primary_workflow_renders_report_sources_review_activity_and_exports(
    app_path: Path,
) -> None:
    engine = FakeEngine()
    app = _app(app_path, engine).run()
    app.text_area(key="research_topic").set_value(
        "How should multi-agent research systems be evaluated?"
    )

    app.button(key="start_research").click().run()

    assert len(engine.requests) == 1
    rendered_text = " ".join(item.value for item in app.markdown)
    assert "Strong evaluation combines evidence quality" in rendered_text
    assert (
        ui_app._escape_markdown_text("A practical guide to evaluated multi-agent research")
        in rendered_text
    )
    assert "Claims are connected to the source library" in rendered_text
    assert "Prepared the final report" in rendered_text
    assert any(r"demo\_fixture" in item.value for item in app.caption)
    assert sum(item.value == "Done" for item in app.caption) == 4
    assert app.text_area(key="follow_up_question")
    assert app.text_area(key="follow_up_question").max_chars == 1_000
    assert app.button(key="ask_follow_up")
    assert len(app.get("download_button")) == 3
    assert not app.exception


@pytest.mark.ui
def test_recent_run_can_be_reopened_from_sidebar(app_path: Path) -> None:
    request = ResearchRequest(topic="How should multi-agent systems be evaluated?")
    previous = _result_for(request, run_id="run-previous")
    app = _app(app_path, FakeEngine(history=(previous,))).run()

    app.button(key="recent_run_run-previous").click().run()

    assert any("Evaluating multi-agent research systems" in item.value for item in app.markdown)
    assert not app.exception


@pytest.mark.ui
def test_provider_source_text_is_rendered_as_literal_text(app_path: Path) -> None:
    request = ResearchRequest(topic="How should multi-agent systems be evaluated?")
    previous = _result_for(request, run_id="run-unsafe-title")
    unsafe_title = "![track](https://evil.example/pixel)"
    previous = previous.model_copy(
        update={
            "sources": (
                previous.sources[0].model_copy(
                    update={
                        "title": unsafe_title,
                        "snippet": unsafe_title,
                        "provider": unsafe_title,
                        "authors": (unsafe_title,),
                    }
                ),
            )
        }
    )
    app = _app(app_path, FakeEngine(history=(previous,))).run()

    app.button(key="recent_run_run-unsafe-title").click().run()

    rendered_values = [item.value for item in (*app.markdown, *app.caption)]
    assert any(r"\!\[track\]\(https://evil\.example/pixel\)" in value for value in rendered_values)
    assert not any(value == unsafe_title for value in rendered_values)
    assert not app.exception


@pytest.mark.ui
def test_generated_report_neutralizes_remote_images_and_raw_html(app_path: Path) -> None:
    request = ResearchRequest(topic="How should multi-agent systems be evaluated?")
    previous = _result_for(request, run_id="run-unsafe-report")
    unsafe_markdown = (
        "# Reviewed report\n\n"
        "![track](https://evil.example/pixel)\n\n"
        "<img src=https://evil.example/second-pixel>\n\n"
        "The remaining report content is source-grounded and long enough to satisfy the "
        "validated report contract while preserving a citation to the registered evidence [S1]."
    )
    assert previous.report is not None
    previous = previous.model_copy(
        update={"report": previous.report.model_copy(update={"markdown": unsafe_markdown})}
    )
    app = _app(app_path, FakeEngine(history=(previous,))).run()

    app.button(key="recent_run_run-unsafe-report").click().run()

    markdown_values = [item.value for item in app.markdown]
    assert any(r"\![track](https://evil.example/pixel)" in value for value in markdown_values)
    assert any(
        "&lt;img src=https://evil.example/second-pixel&gt;" in value for value in markdown_values
    )
    assert not app.exception


@pytest.mark.ui
def test_run_labels_warnings_and_limitations_escape_markdown_injection(app_path: Path) -> None:
    unsafe = "![x](https://evil.example/pixel)"
    request = ResearchRequest(topic=f"Study {unsafe}")
    previous = _result_for(request, run_id="run-unsafe-labels")
    assert previous.report is not None
    previous = previous.model_copy(
        update={
            "warnings": (unsafe,),
            "report": previous.report.model_copy(
                update={
                    "title": f"Report {unsafe}",
                    "executive_summary": (
                        f"Summary {unsafe} remains long enough for the validated report contract."
                    ),
                    "limitations": (unsafe,),
                }
            ),
        }
    )
    app = _app(app_path, FakeEngine(history=(previous,))).run()

    expected_topic = ui_app._escape_markdown_text(request.topic)
    assert any(item.label == expected_topic for item in app.button)
    app.button(key="recent_run_run-unsafe-labels").click().run()

    rendered = [item.value for item in (*app.markdown, *app.warning)]
    assert any(ui_app._escape_markdown_text(f"Report {unsafe}") in value for value in rendered)
    assert any(ui_app._escape_markdown_text(unsafe) in value for value in rendered)
    assert not any(value == unsafe for value in rendered)
    assert not app.exception


@pytest.mark.ui
def test_library_empty_state_is_clear(app_path: Path) -> None:
    app = _app(app_path, FakeEngine()).run()

    app.sidebar.radio(key="workspace_view").set_value("Library").run()

    assert any("No saved sources" in item.value for item in app.info)


@pytest.mark.ui
def test_engine_failure_is_sanitized(app_path: Path) -> None:
    app = _app(app_path, FailingEngine()).run()
    app.text_area(key="research_topic").set_value(
        "How should multi-agent research systems be evaluated?"
    )

    app.button(key="start_research").click().run()

    visible_errors = " ".join(item.value for item in app.error)
    assert "Research stopped before the report was ready" in visible_errors
    assert "provider-secret" not in visible_errors
    assert not app.exception
