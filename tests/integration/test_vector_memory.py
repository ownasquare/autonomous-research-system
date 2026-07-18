from __future__ import annotations

import stat
from hashlib import sha256

from research_system.memory.vector import VectorMemory
from research_system.models import (
    Critique,
    CritiqueDisposition,
    IntegrityLabel,
    ResearchMode,
    ResearchReport,
    Source,
    SourceKind,
)


def _report() -> ResearchReport:
    return ResearchReport(
        run_id="run-memory-1",
        thread_id="thread-memory-1",
        topic="LangGraph supervisor orchestration",
        title="Supervising specialist research agents",
        executive_summary="Supervisor graphs coordinate bounded specialist work and review loops.",
        markdown="# Report\n\n" + "Grounded analysis of supervisor and worker orchestration. " * 5,
        source_ids=("S1",),
        critique=Critique(
            disposition=CritiqueDisposition.APPROVED,
            overall_score=0.9,
            citation_coverage=1.0,
            source_quality=0.8,
        ),
        provenance_mode=ResearchMode.DEMO,
    )


def _source() -> Source:
    content = "A source describing supervisor-worker graph orchestration."
    return Source(
        id="S1",
        kind=SourceKind.DEMO,
        title="Agent graph source",
        url="demo://agent-graph-source",
        content=content,
        provider="bundled demo",
        integrity=IntegrityLabel.DEMO_FIXTURE,
        checksum=sha256(content.encode()).hexdigest(),
    )


def test_vector_memory_survives_reopen(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    memory.remember(_report(), [_source()])

    recalled = VectorMemory(tmp_path).recall("supervisor worker orchestration", limit=1)

    assert stat.S_IMODE(memory.path.stat().st_mode) == 0o600
    assert len(recalled) == 1
    assert recalled[0].kind is SourceKind.MEMORY
    assert recalled[0].integrity is IntegrityLabel.ACCEPTED_MEMORY
    assert recalled[0].locator == "run-memory-1"


def test_remember_is_idempotent_for_a_run(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    memory.remember(_report(), [_source()])
    memory.remember(_report(), [_source()])

    assert len(memory.recall("LangGraph", limit=10)) == 1


def test_recall_ranks_relevant_reports_first(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    graph_report = _report()
    coral_report = _report().model_copy(
        update={
            "run_id": "run-coral",
            "title": "Coral reef biodiversity",
            "topic": "Marine ecosystems",
            "executive_summary": "Coral reefs support diverse marine ecosystems.",
            "markdown": "# Report\n\nCoral reefs, fish, and marine biodiversity.",
        }
    )
    memory.remember(coral_report)
    memory.remember(graph_report)

    recalled = memory.recall("supervisor worker agent orchestration", limit=2)

    assert [source.locator for source in recalled] == ["run-memory-1", "run-coral"]
    assert all(source.score is not None and 0.0 <= source.score <= 1.0 for source in recalled)


def test_recall_ignores_blank_queries(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    memory.remember(_report())

    assert memory.recall("   ") == []


def test_recall_neutralizes_historical_citation_tokens(tmp_path) -> None:
    memory = VectorMemory(tmp_path)
    report = _report().model_copy(
        update={
            "markdown": (
                "# Report\n\nA prior finding used a historical citation [S1]. "
                "This text is long enough to remain a valid stored research report."
            )
        }
    )
    memory.remember(report, [_source()])

    recalled = memory.recall("supervisor orchestration", limit=1)

    assert "[S1]" not in recalled[0].content
    assert "[prior-S1]" in recalled[0].content
    assert "[prior-S1]" in recalled[0].snippet
