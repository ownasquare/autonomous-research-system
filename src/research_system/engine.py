"""Streaming application facade over the supervised research graph."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from threading import RLock
from types import TracebackType
from typing import Any, Protocol, Self
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from research_system.config import Settings, get_settings
from research_system.exceptions import ResearchSystemError
from research_system.graph import ResearchGraph, build_research_graph
from research_system.llm import build_research_model
from research_system.models import (
    AgentTraceEvent,
    ConversationTurn,
    IntegrityLabel,
    ResearchReport,
    ResearchRequest,
    ResearchResult,
    RunStatus,
    Source,
    WorkflowEvent,
    utc_now,
)
from research_system.state import ResearchState


class EngineSourcePipeline(Protocol):
    def parse_uploads(self, uploads: Iterable[object]) -> tuple[list[Source], list[str]]: ...


class RunStore(Protocol):
    def save(self, result: ResearchResult) -> None: ...

    def get(self, run_id: str) -> ResearchResult | None: ...

    def list_runs(self, limit: int = 50) -> list[ResearchResult]: ...

    def close(self) -> None: ...


class ReportMemory(Protocol):
    def remember(self, report: ResearchReport, sources: Sequence[Source]) -> None: ...


class Closeable(Protocol):
    def close(self) -> None: ...


_STAGE_BY_AGENT: dict[str, tuple[str, float]] = {
    "researcher": ("gathering", 0.25),
    "summarizer": ("organizing", 0.5),
    "critic": ("reviewing", 0.75),
    "writer": ("writing", 0.9),
}


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, ResearchSystemError):
        return str(error)
    return "Research workflow failed."


class ResearchEngine:
    """Own workflow lifecycle, streaming events, persistence, and memory readback."""

    def __init__(
        self,
        *,
        graph: ResearchGraph,
        source_pipeline: EngineSourcePipeline,
        repository: RunStore,
        memory: ReportMemory,
        settings: Settings,
        checkpoint_manager: Closeable | None = None,
    ) -> None:
        self.graph = graph
        self.source_pipeline = source_pipeline
        self.repository = repository
        self.memory = memory
        self.settings = settings
        self.checkpoint_manager = checkpoint_manager
        self._run_lock = RLock()
        self._closed = False

    def _initial_result(self, request: ResearchRequest, thread_id: str | None) -> ResearchResult:
        return ResearchResult(
            thread_id=thread_id or str(uuid4()),
            request=request,
            status=RunStatus.PENDING,
        )

    def _state_result(self, pending: ResearchResult, state: dict[str, Any]) -> ResearchResult:
        report = ResearchReport.model_validate(state["report"])
        warnings = tuple(dict.fromkeys(str(item) for item in state.get("warnings", [])))
        status = RunStatus.COMPLETED_WITH_WARNINGS if warnings else RunStatus.COMPLETED
        return ResearchResult(
            run_id=pending.run_id,
            thread_id=pending.thread_id,
            request=pending.request,
            status=status,
            sources=tuple(Source.model_validate(item) for item in state.get("sources", [])),
            report=report,
            trace=tuple(AgentTraceEvent.model_validate(item) for item in state.get("trace", [])),
            conversation=tuple(
                ConversationTurn.model_validate(item) for item in state.get("conversation", [])
            ),
            warnings=warnings,
            created_at=pending.created_at,
            updated_at=utc_now(),
        )

    def _prior_thread_uploads(
        self,
        request: ResearchRequest,
        thread_id: str | None,
    ) -> list[Source]:
        """Recover accepted evidence for a same-thread follow-up when checkpoints exist."""

        if not request.follow_up or not thread_id:
            return []
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = self.graph.get_state(config)
        except ValueError:
            # Graphs without a checkpointer are valid for isolated unit tests.
            return []
        values = snapshot.values if snapshot is not None else {}
        sources = [Source.model_validate(item) for item in values.get("sources", [])]
        return [source for source in sources if source.integrity == IntegrityLabel.USER_UPLOAD]

    @staticmethod
    def _merge_uploaded_sources(*groups: Sequence[Source]) -> list[Source]:
        merged: list[Source] = []
        seen: set[tuple[str, str]] = set()
        for group in groups:
            for source in group:
                key = (source.url, source.checksum)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(source)
        return merged

    def iter_run(
        self,
        request: ResearchRequest,
        thread_id: str | None = None,
        uploads: Iterable[object] = (),
    ) -> Iterator[WorkflowEvent]:
        """Serialize one run across the shared local SQLite-backed engine."""

        if self._closed:
            raise RuntimeError("ResearchEngine is closed")
        with self._run_lock:
            yield from self._iter_run_unlocked(request, thread_id=thread_id, uploads=uploads)

    def _iter_run_unlocked(
        self,
        request: ResearchRequest,
        thread_id: str | None = None,
        uploads: Iterable[object] = (),
    ) -> Iterator[WorkflowEvent]:
        """Run research and yield one stable event per specialist plus a terminal event."""

        if self._closed:
            raise RuntimeError("ResearchEngine is closed")
        pending = self._initial_result(request, thread_id)
        self.repository.save(pending)
        running = pending.model_copy(update={"status": RunStatus.RUNNING, "updated_at": utc_now()})
        self.repository.save(running)
        last_state: dict[str, Any] = {}
        seen_trace = 0
        try:
            parsed_sources, upload_warnings = self.source_pipeline.parse_uploads(tuple(uploads))
            prior_uploads = self._prior_thread_uploads(request, thread_id)
            uploaded_sources = self._merge_uploaded_sources(parsed_sources, prior_uploads)
            user_content = request.follow_up or (
                f"Research topic: {request.topic}\nObjective: {request.objective}"
            )
            initial_state: ResearchState = {
                "run_id": pending.run_id,
                "thread_id": pending.thread_id,
                "request": request.model_dump(mode="json"),
                "uploaded_sources": [source.model_dump(mode="json") for source in uploaded_sources],
                "sources": [],
                "query_plan": None,
                "synthesis": None,
                "critique": None,
                "report": None,
                "route": "",
                "revision_count": 0,
                "trace": [],
                "warnings": upload_warnings,
                "conversation": [
                    ConversationTurn(role="user", content=user_content).model_dump(mode="json")
                ],
                "error": None,
                "provenance_mode": self.settings.research_mode.value,
            }
            config: RunnableConfig = {"configurable": {"thread_id": pending.thread_id}}
            for snapshot in self.graph.stream(initial_state, config, stream_mode="values"):
                last_state = dict(snapshot)
                raw_trace = last_state.get("trace", [])
                for raw_event in raw_trace[seen_trace:]:
                    trace_event = AgentTraceEvent.model_validate(raw_event)
                    stage_progress = _STAGE_BY_AGENT.get(trace_event.agent)
                    if stage_progress is None:
                        continue
                    stage, progress = stage_progress
                    yield WorkflowEvent(
                        run_id=pending.run_id,
                        stage=stage,
                        agent=trace_event.agent,
                        message=trace_event.message,
                        progress=progress,
                    )
                seen_trace = len(raw_trace)

            if not last_state.get("report"):
                raise RuntimeError("Workflow ended without a report")
            result = self._state_result(pending, last_state)
            if result.report is None:
                raise RuntimeError("Workflow result did not validate a report")
            if (
                result.status == RunStatus.COMPLETED
                and result.report.critique.disposition.value == "approved"
            ):
                try:
                    self.memory.remember(result.report, result.sources)
                except Exception:
                    warnings = (*result.warnings, "Long-term memory could not be updated.")
                    result = result.model_copy(
                        update={"warnings": warnings, "status": RunStatus.COMPLETED_WITH_WARNINGS}
                    )
            self.repository.save(result)
            yield WorkflowEvent(
                run_id=pending.run_id,
                stage="complete",
                agent="supervisor",
                message="Research report completed and validated.",
                progress=1.0,
                result=result,
            )
        except Exception as error:
            message = _safe_error_message(error)
            trace = tuple(
                AgentTraceEvent.model_validate(item) for item in last_state.get("trace", [])
            )
            trace = (
                *trace,
                AgentTraceEvent(
                    sequence=len(trace) + 1,
                    agent="supervisor",
                    status="failed",
                    message=message,
                ),
            )
            failed = ResearchResult(
                run_id=pending.run_id,
                thread_id=pending.thread_id,
                request=request,
                status=RunStatus.FAILED,
                sources=tuple(
                    Source.model_validate(item) for item in last_state.get("sources", [])
                ),
                trace=trace,
                warnings=tuple(str(item) for item in last_state.get("warnings", [])),
                error=message,
                created_at=pending.created_at,
                updated_at=utc_now(),
            )
            self.repository.save(failed)
            yield WorkflowEvent(
                run_id=pending.run_id,
                stage="failed",
                agent="supervisor",
                message=message,
                progress=1.0,
                result=failed,
            )

    def run(
        self,
        request: ResearchRequest,
        thread_id: str | None = None,
        uploads: Iterable[object] = (),
    ) -> ResearchResult:
        """Run research synchronously and return its terminal validated result."""

        terminal: ResearchResult | None = None
        for event in self.iter_run(request, thread_id=thread_id, uploads=uploads):
            if event.result is not None:
                terminal = event.result
        if terminal is None:
            raise RuntimeError("Research workflow emitted no terminal result")
        return terminal

    def get_run(self, run_id: str) -> ResearchResult | None:
        return self.repository.get(run_id)

    def list_runs(self, limit: int = 50) -> list[ResearchResult]:
        return self.repository.list_runs(limit=limit)

    def close(self) -> None:
        with self._run_lock:
            if self._closed:
                return
            try:
                self.repository.close()
            finally:
                if self.checkpoint_manager is not None:
                    self.checkpoint_manager.close()
                self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def build_default_engine(settings: Settings | None = None) -> ResearchEngine:
    """Build the complete local engine with durable memory and checkpoints."""

    from research_system.memory.vector import VectorMemory
    from research_system.persistence.checkpoints import SqliteCheckpointManager
    from research_system.persistence.runs import RunRepository
    from research_system.tools.sources import SourcePipeline

    resolved_settings = settings or get_settings()
    resolved_settings.ensure_directories()
    memory = VectorMemory(resolved_settings.vector_path)
    pipeline = SourcePipeline(resolved_settings, memory=memory)
    repository = RunRepository(resolved_settings.runs_path)
    checkpoint_manager = SqliteCheckpointManager(resolved_settings.checkpoint_path)
    graph = build_research_graph(
        build_research_model(resolved_settings),
        pipeline,
        checkpointer=checkpoint_manager.saver,
    )
    return ResearchEngine(
        graph=graph,
        source_pipeline=pipeline,
        repository=repository,
        memory=memory,
        settings=resolved_settings,
        checkpoint_manager=checkpoint_manager,
    )
