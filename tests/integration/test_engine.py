from __future__ import annotations

from hashlib import sha256

from langgraph.checkpoint.memory import InMemorySaver

from research_system.config import Settings
from research_system.engine import ResearchEngine, build_default_engine
from research_system.graph import build_research_graph
from research_system.llm import DeterministicResearchModel
from research_system.models import (
    Critique,
    CritiqueDisposition,
    IntegrityLabel,
    ResearchRequest,
    RunStatus,
    Source,
    SourceKind,
)


def _source() -> Source:
    content = (
        "A persistent workflow trace makes multi-agent research inspectable from evidence "
        "gathering through the final citation check. The ZephyrContinuity principle marks "
        "the prior report's distinctive recommendation."
    )
    return Source(
        id="S1",
        kind=SourceKind.DEMO,
        title="Persistent research traces",
        url="demo://persistent-traces",
        content=content,
        provider="bundled-demo",
        integrity=IntegrityLabel.DEMO_FIXTURE,
        checksum=sha256(content.encode()).hexdigest(),
    )


def _uploaded_source() -> Source:
    content = (
        "Uploaded evaluation evidence establishes the ZephyrUpload benchmark as the "
        "decision anchor for a source-preserving research follow-up."
    )
    return Source(
        id="S1",
        kind=SourceKind.PDF,
        title="Uploaded evaluation evidence",
        url="upload://evaluation-evidence.pdf",
        content=content,
        provider="user-upload",
        integrity=IntegrityLabel.USER_UPLOAD,
        checksum=sha256(content.encode()).hexdigest(),
    )


class Pipeline:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.uploads: tuple[object, ...] = ()
        self.query_batches: list[tuple[str, ...]] = []
        self.uploaded_batches: list[tuple[Source, ...]] = []

    def parse_uploads(self, uploads):
        self.uploads = tuple(uploads)
        return [], []

    def gather(self, request, queries, uploaded_sources=()):
        del request
        self.query_batches.append(tuple(queries))
        self.uploaded_batches.append(tuple(uploaded_sources))
        if self.fail:
            raise RuntimeError("provider internals must stay private")
        return [_source()], []


class UploadOnlyPipeline:
    def __init__(self, *, warn_first: bool = False) -> None:
        self.warn_first = warn_first
        self.calls = 0
        self.gathered_uploads: list[tuple[Source, ...]] = []

    def parse_uploads(self, uploads):
        return ([_uploaded_source()] if tuple(uploads) else []), []

    def gather(self, request, queries, uploaded_sources=()):
        del request, queries
        self.calls += 1
        sources = tuple(uploaded_sources)
        self.gathered_uploads.append(sources)
        warnings = ["First-run provider warning."] if self.warn_first and self.calls == 1 else []
        return list(sources), warnings


class Repository:
    def __init__(self) -> None:
        self.records = {}
        self.saved_statuses = []
        self.closed = False

    def save(self, result):
        self.records[result.run_id] = result
        self.saved_statuses.append(result.status)

    def get(self, run_id):
        return self.records.get(run_id)

    def list_runs(self, limit=50):
        return list(self.records.values())[-limit:]

    def close(self):
        self.closed = True


class Memory:
    def __init__(self) -> None:
        self.remembered = []

    def remember(self, report, sources):
        self.remembered.append((report, tuple(sources)))


class CheckpointManager:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class RevisionRequiredModel(DeterministicResearchModel):
    def critique(self, request, sources, synthesis):
        del request, sources, synthesis
        return Critique(
            disposition=CritiqueDisposition.REVISE_SUMMARY,
            overall_score=0.6,
            citation_coverage=1.0,
            source_quality=0.7,
            gaps=("Clarify the decision boundary before approval.",),
        )


def _engine(tmp_path, *, fail: bool = False, model=None):
    pipeline = Pipeline(fail=fail)
    repository = Repository()
    memory = Memory()
    graph = build_research_graph(model or DeterministicResearchModel(), pipeline)
    engine = ResearchEngine(
        graph=graph,
        source_pipeline=pipeline,
        repository=repository,
        memory=memory,
        settings=Settings(research_data_dir=tmp_path),
    )
    return engine, pipeline, repository, memory


def test_engine_streams_stable_stages_and_persists_result(tmp_path) -> None:
    engine, pipeline, repository, memory = _engine(tmp_path)
    request = ResearchRequest(topic="How should multi-agent research systems be evaluated?")

    events = list(engine.iter_run(request, thread_id="thread-1", uploads=(b"pdf",)))

    assert [event.stage for event in events] == [
        "gathering",
        "organizing",
        "reviewing",
        "writing",
        "complete",
    ]
    result = events[-1].result
    assert result is not None
    assert result.status == RunStatus.COMPLETED
    assert result.thread_id == "thread-1"
    assert repository.get(result.run_id) == result
    assert repository.saved_statuses[:2] == [RunStatus.PENDING, RunStatus.RUNNING]
    assert memory.remembered[0][0] == result.report
    assert pipeline.uploads == (b"pdf",)


def test_engine_marks_exhausted_critic_revision_as_completed_with_warnings(tmp_path) -> None:
    engine, _, repository, memory = _engine(tmp_path, model=RevisionRequiredModel())

    result = engine.run(
        ResearchRequest(
            topic="How should multi-agent research systems be evaluated?",
            max_revisions=0,
        )
    )

    assert result.status == RunStatus.COMPLETED_WITH_WARNINGS
    assert result.report is not None
    assert result.report.critique.disposition == CritiqueDisposition.REVISE_SUMMARY
    assert any("revision budget (0) was exhausted" in warning for warning in result.warnings)
    assert repository.get(result.run_id) == result
    assert memory.remembered == []


def test_engine_returns_sanitized_failed_result(tmp_path) -> None:
    engine, _, repository, _ = _engine(tmp_path, fail=True)

    result = engine.run(
        ResearchRequest(topic="How should multi-agent research systems be evaluated?")
    )

    assert result.status == RunStatus.FAILED
    assert result.error == "Research workflow failed."
    assert "provider internals" not in result.error
    assert repository.get(result.run_id) == result


def test_engine_readback_delegates_to_repository(tmp_path) -> None:
    engine, _, _, _ = _engine(tmp_path)
    result = engine.run(
        ResearchRequest(topic="How should multi-agent research systems be evaluated?")
    )

    assert engine.get_run(result.run_id) == result
    assert engine.list_runs(limit=1) == [result]


def test_engine_close_releases_repository_and_checkpoint_connections(tmp_path) -> None:
    engine, _, repository, _ = _engine(tmp_path)
    checkpoints = CheckpointManager()
    engine.checkpoint_manager = checkpoints

    engine.close()
    engine.close()

    assert repository.closed is True
    assert checkpoints.closed is True


def test_default_engine_preserves_short_term_conversation_by_thread(tmp_path) -> None:
    settings = Settings(research_data_dir=tmp_path)
    topic = "How should multi-agent research systems be evaluated?"

    with build_default_engine(settings) as engine:
        first = engine.run(ResearchRequest(topic=topic), thread_id="thread-memory")
        second = engine.run(
            ResearchRequest(
                topic=topic,
                follow_up="Compare the findings with the critic's quality criteria.",
            ),
            thread_id="thread-memory",
        )

    assert first.status == RunStatus.COMPLETED
    assert second.status == RunStatus.COMPLETED
    assert [turn.role for turn in second.conversation] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert second.conversation[-2].content.startswith("Compare the findings")


def test_follow_up_consumes_prior_report_in_query_plan_and_output(tmp_path) -> None:
    pipeline = Pipeline()
    repository = Repository()
    graph = build_research_graph(
        DeterministicResearchModel(), pipeline, checkpointer=InMemorySaver()
    )
    engine = ResearchEngine(
        graph=graph,
        source_pipeline=pipeline,
        repository=repository,
        memory=Memory(),
        settings=Settings(research_data_dir=tmp_path),
    )
    topic = "How should multi-agent research systems be evaluated?"

    first = engine.run(ResearchRequest(topic=topic), thread_id="thread-context")
    second = engine.run(
        ResearchRequest(
            topic=topic,
            follow_up="Which earlier recommendation should drive the next evaluation?",
        ),
        thread_id="thread-context",
    )

    assert first.status == RunStatus.COMPLETED
    assert second.status == RunStatus.COMPLETED
    assert not any("zephyrcontinuity" in query.casefold() for query in pipeline.query_batches[0])
    assert any("zephyrcontinuity" in query.casefold() for query in pipeline.query_batches[1])
    assert pipeline.uploaded_batches == [(), ()]
    assert second.report is not None
    assert "## Follow-up continuity" in second.report.markdown
    assert "`zephyrcontinuity`" in second.report.markdown.casefold()


def test_follow_up_reuses_prior_uploaded_evidence_without_reupload(tmp_path) -> None:
    pipeline = UploadOnlyPipeline()
    graph = build_research_graph(
        DeterministicResearchModel(), pipeline, checkpointer=InMemorySaver()
    )
    engine = ResearchEngine(
        graph=graph,
        source_pipeline=pipeline,
        repository=Repository(),
        memory=Memory(),
        settings=Settings(research_data_dir=tmp_path),
    )
    topic = "How should uploaded evaluation evidence guide an agent benchmark?"

    first = engine.run(
        ResearchRequest(topic=topic, use_web=False, use_arxiv=False, use_memory=False),
        thread_id="thread-upload",
        uploads=(b"pdf",),
    )
    second = engine.run(
        ResearchRequest(
            topic=topic,
            follow_up="Which uploaded benchmark should anchor the next decision?",
            use_web=False,
            use_arxiv=False,
            use_memory=False,
        ),
        thread_id="thread-upload",
    )

    assert first.status == RunStatus.COMPLETED
    assert second.status == RunStatus.COMPLETED
    assert pipeline.gathered_uploads[0][0].url == "upload://evaluation-evidence.pdf"
    assert pipeline.gathered_uploads[1][0].url == "upload://evaluation-evidence.pdf"
    assert second.sources[0].integrity == IntegrityLabel.USER_UPLOAD
    assert second.report is not None
    assert "zephyrupload" in second.report.markdown.casefold()


def test_follow_up_resets_prior_run_warnings(tmp_path) -> None:
    pipeline = UploadOnlyPipeline(warn_first=True)
    graph = build_research_graph(
        DeterministicResearchModel(), pipeline, checkpointer=InMemorySaver()
    )
    memory = Memory()
    engine = ResearchEngine(
        graph=graph,
        source_pipeline=pipeline,
        repository=Repository(),
        memory=memory,
        settings=Settings(research_data_dir=tmp_path),
    )
    topic = "How should uploaded evaluation evidence guide an agent benchmark?"

    first = engine.run(
        ResearchRequest(topic=topic, use_web=False, use_arxiv=False, use_memory=False),
        thread_id="thread-warning-reset",
        uploads=(b"pdf",),
    )
    second = engine.run(
        ResearchRequest(
            topic=topic,
            follow_up="Which recommendation remains strongest?",
            use_web=False,
            use_arxiv=False,
            use_memory=False,
        ),
        thread_id="thread-warning-reset",
    )

    assert first.status == RunStatus.COMPLETED_WITH_WARNINGS
    assert first.warnings == ("First-run provider warning.",)
    assert second.status == RunStatus.COMPLETED
    assert second.warnings == ()
    assert len(memory.remembered) == 1
