"""Streamlit workbench for supervised research runs."""

from __future__ import annotations

import atexit
import html
import re
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

import streamlit as st

from research_system.models import (
    ResearchDepth,
    ResearchMode,
    ResearchRequest,
    ResearchResult,
    RunStatus,
    Source,
    WorkflowEvent,
)
from research_system.ui.components import (
    downloadable_filename,
    fallback_html_export,
    fallback_json_export,
    fallback_markdown_export,
    stage_label,
    stage_progress,
)
from research_system.ui.theme import inject_theme


class ResearchEngine(Protocol):
    """Small engine surface consumed by the workbench."""

    def iter_run(
        self,
        request: ResearchRequest,
        thread_id: str | None = None,
        uploads: Sequence[tuple[str, bytes]] = (),
    ) -> Iterator[WorkflowEvent]: ...

    def run(
        self,
        request: ResearchRequest,
        thread_id: str | None = None,
        uploads: Sequence[tuple[str, bytes]] = (),
    ) -> ResearchResult: ...

    def get_run(self, run_id: str) -> ResearchResult | None: ...

    def list_runs(self, limit: int = 20) -> list[ResearchResult]: ...

    def close(self) -> None: ...


EngineFactory = Callable[[], ResearchEngine]

_ENGINE_FACTORY_OVERRIDE: EngineFactory | None = None
_ENGINE_STATE_KEY = "_research_desk_engine"
_ENGINE_ERROR_KEY = "_research_desk_engine_error"
_ACTIVE_RESULT_KEY = "active_research_result"
_EVENTS_KEY = "research_workflow_events"
_MARKDOWN_CONTROL_PATTERN = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|])")
_PROCESS_ENGINE: ResearchEngine | None = None
_PROCESS_ENGINE_ERROR: str | None = None
_PROCESS_ENGINE_LOCK = Lock()


def _escape_markdown_text(value: str) -> str:
    """Render untrusted provider/model text literally inside Markdown containers."""

    return _MARKDOWN_CONTROL_PATTERN.sub(r"\\\1", value)


def _sanitize_report_markdown(value: str) -> str:
    """Preserve report structure while neutralizing raw HTML and remote images."""

    escaped_html = html.escape(value, quote=False)
    return escaped_html.replace("![", r"\![")


def set_engine_factory_for_testing(factory: EngineFactory) -> None:
    """Install a deterministic backend factory for Streamlit AppTest."""

    global _ENGINE_FACTORY_OVERRIDE
    _ENGINE_FACTORY_OVERRIDE = factory


def clear_engine_factory_for_testing() -> None:
    """Remove the AppTest backend override."""

    global _ENGINE_FACTORY_OVERRIDE
    _ENGINE_FACTORY_OVERRIDE = None


def _default_engine_factory() -> ResearchEngine:
    from research_system.engine import build_default_engine

    return cast(ResearchEngine, build_default_engine())


def _get_process_engine() -> tuple[ResearchEngine | None, str | None]:
    """Create exactly one default engine for every local Streamlit process."""

    global _PROCESS_ENGINE, _PROCESS_ENGINE_ERROR
    with _PROCESS_ENGINE_LOCK:
        if _PROCESS_ENGINE is not None:
            return _PROCESS_ENGINE, None
        if _PROCESS_ENGINE_ERROR is not None:
            return None, _PROCESS_ENGINE_ERROR
        try:
            _PROCESS_ENGINE = _default_engine_factory()
        except Exception:
            _PROCESS_ENGINE_ERROR = (
                "The research service is not ready yet. You can review this workspace, "
                "but starting a run is temporarily unavailable."
            )
            return None, _PROCESS_ENGINE_ERROR
        return _PROCESS_ENGINE, None


def _close_process_engine() -> None:
    """Release the process-owned engine once, including during interpreter shutdown."""

    global _PROCESS_ENGINE
    with _PROCESS_ENGINE_LOCK:
        engine = _PROCESS_ENGINE
        _PROCESS_ENGINE = None
    if engine is not None:
        engine.close()


atexit.register(_close_process_engine)


def _get_engine() -> tuple[ResearchEngine | None, str | None]:
    if _ENGINE_FACTORY_OVERRIDE is None:
        return _get_process_engine()
    if _ENGINE_STATE_KEY in st.session_state:
        return cast(ResearchEngine, st.session_state[_ENGINE_STATE_KEY]), None
    if st.session_state.get(_ENGINE_ERROR_KEY):
        return None, cast(str, st.session_state[_ENGINE_ERROR_KEY])

    factory = _ENGINE_FACTORY_OVERRIDE
    try:
        engine = factory()
    except Exception:
        message = (
            "The research service is not ready yet. You can review this workspace, "
            "but starting a run is temporarily unavailable."
        )
        st.session_state[_ENGINE_ERROR_KEY] = message
        return None, message

    st.session_state[_ENGINE_STATE_KEY] = engine
    return engine, None


def _safe_history(engine: ResearchEngine | None, *, limit: int = 20) -> list[ResearchResult]:
    if engine is None:
        return []
    try:
        return engine.list_runs(limit=limit)
    except Exception:
        return []


def _engine_mode(engine: ResearchEngine | None) -> ResearchMode | None:
    settings = getattr(engine, "settings", None)
    raw_mode = getattr(settings, "research_mode", None)
    if isinstance(raw_mode, ResearchMode):
        return raw_mode
    if isinstance(raw_mode, str):
        try:
            return ResearchMode(raw_mode)
        except ValueError:
            return None
    return None


def _upload_limit_megabytes(engine: ResearchEngine | None) -> int:
    settings = getattr(engine, "settings", None)
    raw_limit = getattr(settings, "max_pdf_bytes", 15_000_000)
    if not isinstance(raw_limit, int) or raw_limit <= 0:
        return 15
    return max(1, raw_limit // 1_000_000)


def _upload_batch_limits(engine: ResearchEngine | None) -> tuple[int, int]:
    settings = getattr(engine, "settings", None)
    raw_count = getattr(settings, "max_pdf_uploads", 5)
    raw_total = getattr(settings, "max_upload_bytes_total", 30_000_000)
    count = raw_count if isinstance(raw_count, int) and raw_count > 0 else 5
    total = raw_total if isinstance(raw_total, int) and raw_total > 0 else 30_000_000
    return count, max(1, total // 1_000_000)


def _open_saved_run(engine: ResearchEngine, run_id: str) -> None:
    try:
        result = engine.get_run(run_id)
    except Exception:
        result = None
    if result is not None:
        st.session_state[_ACTIVE_RESULT_KEY] = result
        st.session_state[_EVENTS_KEY] = []
        st.session_state["workspace_view"] = "New research"


def _short_topic(result: ResearchResult, *, length: int = 42) -> str:
    topic = result.request.topic
    shortened = topic if len(topic) <= length else f"{topic[: length - 1].rstrip()}…"
    return _escape_markdown_text(shortened)


def _render_sidebar(
    engine: ResearchEngine | None,
    history: Sequence[ResearchResult],
    engine_error: str | None,
) -> None:
    with st.sidebar:
        st.markdown("### Research Desk")
        st.caption("Source-grounded reports, organized for review.")
        st.radio(
            "Workspace",
            ("New research", "Library", "Settings"),
            key="workspace_view",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("#### Recent runs")
        if history and engine is not None:
            for result in history[:8]:
                st.button(
                    _short_topic(result),
                    key=f"recent_run_{result.run_id}",
                    help=f"Open {_short_topic(result, length=80)}",
                    on_click=_open_saved_run,
                    args=(engine, result.run_id),
                    use_container_width=True,
                )
        else:
            st.caption("Completed research will appear here.")

        st.divider()
        if engine_error:
            st.caption("Research service unavailable")
        else:
            st.caption("Workspace ready")


def _render_stage_timeline(events: Sequence[WorkflowEvent]) -> None:
    st.markdown("#### Research stages")
    active_stage = events[-1].stage if events else ""
    stages = ("gathering", "organizing", "reviewing", "writing")
    active_index = stages.index(active_stage) if active_stage in stages else -1
    complete = active_stage == "complete"

    columns = st.columns(4)
    for index, (column, stage) in enumerate(zip(columns, stages, strict=True)):
        with column:
            st.caption(f"Step {index + 1}")
            st.markdown(f"**{stage_label(stage)}**")
            if complete or index < active_index:
                state = "Done"
            elif index == active_index:
                state = "In progress"
            else:
                state = "Waiting"
            st.caption(state)


def _validate_inputs(
    topic: str,
    sources: Sequence[str],
    uploads: Sequence[object],
) -> str | None:
    if len(topic.strip()) < 5:
        return "Add a research topic with at least five characters."
    if not sources and not uploads:
        return "Choose at least one source type or attach a PDF."
    return None


def _request_from_inputs(
    *,
    topic: str,
    objective: str,
    audience: str,
    depth: str,
    sources: Sequence[str],
) -> ResearchRequest:
    depth_value = ResearchDepth(depth.lower())
    return ResearchRequest(
        topic=topic,
        objective=objective,
        audience=audience,
        depth=depth_value,
        use_web="Web" in sources,
        use_arxiv="arXiv" in sources,
        use_memory="Research memory" in sources,
        use_demo="Bundled demo" in sources,
    )


def _read_uploads(uploaded_files: Sequence[object]) -> tuple[tuple[str, bytes], ...]:
    uploads: list[tuple[str, bytes]] = []
    for uploaded_file in uploaded_files:
        name = Path(str(getattr(uploaded_file, "name", "research.pdf"))).name
        getvalue = getattr(uploaded_file, "getvalue", None)
        if not callable(getvalue):
            continue
        content = getvalue()
        if isinstance(content, bytes):
            uploads.append((name, content))
    return tuple(uploads)


def _execute_run(
    engine: ResearchEngine,
    request: ResearchRequest,
    *,
    uploads: Sequence[tuple[str, bytes]] = (),
    thread_id: str | None = None,
) -> ResearchResult | None:
    events: list[WorkflowEvent] = []
    result: ResearchResult | None = None
    last_run_id: str | None = None
    progress_slot = st.empty()
    message_slot = st.empty()

    try:
        for event in engine.iter_run(request, thread_id=thread_id, uploads=uploads):
            events.append(event)
            last_run_id = event.run_id
            progress_slot.progress(
                event.progress or stage_progress(event.stage),
                text=stage_label(event.stage),
            )
            message_slot.caption(event.message)
            if event.result is not None:
                result = event.result

        if result is None and last_run_id is not None:
            result = engine.get_run(last_run_id)
        if result is None:
            raise RuntimeError("research stream ended without a persisted result")
    except Exception:
        st.session_state[_EVENTS_KEY] = events
        st.error(
            "Research stopped before the report was ready. Your topic is still here, "
            "so you can try again when the service is available."
        )
        return None

    st.session_state[_EVENTS_KEY] = events
    st.session_state[_ACTIVE_RESULT_KEY] = result
    if result.status == RunStatus.FAILED:
        st.error(
            "Research stopped before the report was ready. "
            "Review your source choices and try again."
        )
        return result

    progress_slot.progress(1.0, text="Research complete")
    message_slot.caption("The report is ready to review.")
    return result


def _export_payloads(result: ResearchResult) -> tuple[str, str, str]:
    try:
        from research_system.exports import export_html, export_json, export_markdown

        return export_markdown(result), export_json(result), export_html(result)
    except Exception:
        return (
            fallback_markdown_export(result),
            fallback_json_export(result),
            fallback_html_export(result),
        )


def _render_sources(result: ResearchResult) -> None:
    if not result.sources:
        st.info("No source records were saved for this run.")
        return

    for source in result.sources:
        safe_title = _escape_markdown_text(source.title)
        with st.expander(f"{source.id} · {safe_title}"):
            st.markdown(f"**{safe_title}**")
            metadata = f"{source.kind.value.upper()} · {source.integrity.value} · {source.provider}"
            st.caption(_escape_markdown_text(metadata))
            if source.snippet:
                st.markdown(_escape_markdown_text(source.snippet))
            if source.authors:
                st.caption(_escape_markdown_text(f"Authors: {', '.join(source.authors)}"))
            if source.url.startswith(("https://", "http://")):
                st.link_button("Open source", source.url)
            else:
                st.caption("Available inside this research workspace")


def _render_critique(result: ResearchResult) -> None:
    if result.report is None:
        st.info("A review will appear after the report is written.")
        return
    critique = result.report.critique
    metrics = st.columns(3)
    metrics[0].metric("Overall", f"{critique.overall_score:.0%}")
    metrics[1].metric("Citation coverage", f"{critique.citation_coverage:.0%}")
    metrics[2].metric("Source quality", f"{critique.source_quality:.0%}")

    st.markdown("#### What worked")
    if critique.strengths:
        for strength in critique.strengths:
            st.markdown(f"- {_escape_markdown_text(strength)}")
    else:
        st.caption("No strengths were recorded.")

    st.markdown("#### What to watch")
    notes = (*critique.gaps, *critique.unsupported_claims)
    if notes:
        for note in notes:
            st.markdown(f"- {_escape_markdown_text(note)}")
    else:
        st.caption("No material evidence gaps were recorded.")


def _render_activity(result: ResearchResult) -> None:
    if not result.trace:
        st.info("Activity will appear after the first research stage begins.")
        return
    for event in result.trace:
        st.markdown(f"**{event.agent.title()} · {event.status.title()}**")
        st.markdown(_escape_markdown_text(event.message))
        st.caption(event.created_at.strftime("%b %d, %Y · %H:%M UTC"))


def _render_downloads(result: ResearchResult) -> None:
    markdown_data, json_data, html_data = _export_payloads(result)
    st.markdown("#### Export")
    st.caption("Save a review copy in the format that fits your workflow.")
    columns = st.columns(3)
    columns[0].download_button(
        "Markdown",
        data=markdown_data,
        file_name=downloadable_filename(result.run_id, "markdown"),
        mime="text/markdown",
        key="download_markdown",
        use_container_width=True,
    )
    columns[1].download_button(
        "JSON",
        data=json_data,
        file_name=downloadable_filename(result.run_id, "json"),
        mime="application/json",
        key="download_json",
        use_container_width=True,
    )
    columns[2].download_button(
        "HTML",
        data=html_data,
        file_name=downloadable_filename(result.run_id, "html"),
        mime="text/html",
        key="download_html",
        use_container_width=True,
    )


def _render_follow_up(engine: ResearchEngine | None, result: ResearchResult) -> None:
    st.markdown("#### Ask a follow-up")
    st.caption("Keep the same research thread and ask for a tighter comparison or explanation.")
    follow_up = st.text_area(
        "Follow-up question",
        key="follow_up_question",
        placeholder="For example: Compare the two strongest approaches in a decision table.",
        max_chars=1_000,
        label_visibility="collapsed",
    )
    ask = st.button(
        "Ask follow-up",
        key="ask_follow_up",
        type="secondary",
        disabled=engine is None,
    )
    if not ask or engine is None:
        return
    if len(follow_up.strip()) < 5:
        st.error("Add a follow-up question with at least five characters.")
        return

    try:
        follow_up_request = ResearchRequest.model_validate(
            {**result.request.model_dump(), "follow_up": follow_up.strip()}
        )
    except ValueError:
        st.error("Keep the follow-up question within 1,000 characters.")
        return
    follow_up_result = _execute_run(
        engine,
        follow_up_request,
        thread_id=result.thread_id,
    )
    if follow_up_result is not None:
        st.rerun()


def _render_result(engine: ResearchEngine | None, result: ResearchResult) -> None:
    if result.status == RunStatus.FAILED:
        st.error("This run stopped before a report was produced.")
        return
    if result.report is None:
        st.info("This run is still being prepared. Reopen it in a moment.")
        return

    if result.warnings:
        st.warning(_escape_markdown_text("Completed with notes: " + " ".join(result.warnings)))

    st.divider()
    st.caption(f"Completed research · {len(result.sources)} saved sources")
    st.markdown(f"## {_escape_markdown_text(result.report.title)}")
    tabs = st.tabs(("Report", "Sources", "Critic review", "Activity"))
    with tabs[0]:
        st.markdown("### Executive summary")
        st.markdown(_escape_markdown_text(result.report.executive_summary))
        st.markdown("### Full report")
        st.markdown(_sanitize_report_markdown(result.report.markdown))
        if result.report.limitations:
            with st.expander("Limitations"):
                for limitation in result.report.limitations:
                    st.markdown(f"- {_escape_markdown_text(limitation)}")
    with tabs[1]:
        st.markdown("### Source library")
        _render_sources(result)
    with tabs[2]:
        st.markdown("### Critic review")
        _render_critique(result)
    with tabs[3]:
        st.markdown("### Activity")
        _render_activity(result)

    _render_follow_up(engine, result)
    _render_downloads(result)


def _render_new_research(
    engine: ResearchEngine | None,
    engine_error: str | None,
) -> None:
    st.title("Research Desk")
    st.markdown("Turn a focused question into a reviewable report with a traceable source library.")

    if engine_error:
        st.error(engine_error)

    with st.container(border=True):
        st.markdown("### Start new research")
        engine_mode = _engine_mode(engine)
        if engine_mode is ResearchMode.DEMO:
            st.info(
                "Demo mode - uses bundled fixture evidence; live providers are not being queried."
            )
        topic = st.text_area(
            "Research topic",
            key="research_topic",
            placeholder="What should the research team investigate?",
            height=112,
        )
        objective = st.text_input(
            "Objective",
            value="Produce a balanced, decision-ready research report.",
            key="research_objective",
        )
        audience = st.text_input(
            "Audience",
            value="General professional reader",
            key="research_audience",
        )

        left, right = st.columns(2)
        with left:
            depth = st.selectbox(
                "Depth",
                ("Quick", "Standard", "Deep"),
                index=1,
                key="research_depth",
                help="Quick is concise; Deep uses a broader source and review budget.",
            )
        with right:
            source_options: tuple[str, ...]
            default_sources: list[str]
            if engine_mode is ResearchMode.DEMO:
                source_options = ("Bundled demo", "Research memory")
                default_sources = ["Bundled demo"]
            else:
                source_options = ("Web", "arXiv", "Research memory")
                default_sources = ["Web", "arXiv"]
                if st.session_state.get("default_research_memory", True):
                    default_sources.append("Research memory")
            sources = st.multiselect(
                "Sources",
                source_options,
                default=default_sources,
                key="research_sources",
            )

        upload_count, upload_total_mb = _upload_batch_limits(engine)
        uploaded_files = st.file_uploader(
            "Add PDFs",
            type=("pdf",),
            accept_multiple_files=True,
            key="research_pdfs",
            help=(
                f"Optional. Up to {upload_count} PDFs, "
                f"{_upload_limit_megabytes(engine)} MB each, {upload_total_mb} MB total."
            ),
        )
        start = st.button(
            "Start research",
            key="start_research",
            type="primary",
            disabled=engine is None,
        )

    with st.expander("Details", expanded=False):
        st.markdown("**Workflow**")
        st.code("Gathering → Organizing → Reviewing → Writing", language=None)
        st.markdown(
            "A supervisor coordinates specialist workers. Provider selection and live/demo mode "
            "come from local application settings; credential values are never shown here."
        )

    events = cast(list[WorkflowEvent], st.session_state.get(_EVENTS_KEY, []))
    _render_stage_timeline(events)

    if start and engine is not None:
        validation_error = _validate_inputs(topic, sources, uploaded_files)
        if validation_error:
            st.error(validation_error)
        else:
            try:
                request = _request_from_inputs(
                    topic=topic,
                    objective=objective,
                    audience=audience,
                    depth=depth,
                    sources=sources,
                )
            except ValueError:
                st.error("Review the topic, objective, and audience before starting research.")
            else:
                completed_result = _execute_run(
                    engine,
                    request,
                    uploads=_read_uploads(uploaded_files),
                )
                if completed_result is not None:
                    st.rerun()

    result = cast(ResearchResult | None, st.session_state.get(_ACTIVE_RESULT_KEY))
    if result is None:
        st.info("No report yet. Start with a focused topic and the sources you want to include.")
    else:
        _render_result(engine, result)


def _render_library(history: Sequence[ResearchResult]) -> None:
    st.title("Source library")
    st.markdown("Review the evidence saved with your recent research runs.")

    unique_sources: dict[str, Source] = {}
    for result in history:
        for source in result.sources:
            unique_sources.setdefault(source.checksum, source)

    if not unique_sources:
        st.info("No saved sources yet. Complete a research run to build your library.")
        return

    st.caption(f"{len(unique_sources)} sources across {len(history)} recent runs")
    for source in unique_sources.values():
        with st.container(border=True):
            st.markdown(f"**{_escape_markdown_text(source.title)}**")
            metadata = (
                f"{source.kind.value.upper()} · {source.integrity.value} · "
                f"{source.provider} · {source.id}"
            )
            st.caption(_escape_markdown_text(metadata))
            if source.snippet:
                st.markdown(_escape_markdown_text(source.snippet))
            if source.url.startswith(("https://", "http://")):
                st.link_button("Open source", source.url)


def _render_settings(engine_error: str | None) -> None:
    st.title("Settings")
    st.markdown("Set workspace preferences without exposing provider credentials.")
    with st.container(border=True):
        st.markdown("### Research defaults")
        st.checkbox(
            "Include saved research memory by default",
            value=True,
            key="default_research_memory",
        )
        st.caption("You can still change the source mix for every new run.")

    with st.container(border=True):
        st.markdown("### Service status")
        if engine_error:
            st.error("The research service is unavailable. Check the local setup and try again.")
        else:
            st.success("The research service is ready.")

    with st.expander("Details", expanded=False):
        st.markdown(
            "Research mode, search providers, model selection, and storage locations are managed "
            "through local application settings. This screen never displays secret values."
        )


def render_app() -> None:
    """Render the complete Research Desk application."""

    st.set_page_config(
        page_title="Research Desk",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_theme()
    engine, engine_error = _get_engine()
    history = _safe_history(engine) or []
    _render_sidebar(engine, history, engine_error)

    view = cast(str, st.session_state.get("workspace_view", "New research"))
    if view == "Library":
        _render_library(history)
    elif view == "Settings":
        _render_settings(engine_error)
    else:
        _render_new_research(engine, engine_error)
