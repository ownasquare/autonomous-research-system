from __future__ import annotations

from hashlib import sha256

from research_system.graph import build_research_graph, graph_mermaid
from research_system.llm import DeterministicResearchModel
from research_system.models import (
    Critique,
    CritiqueDisposition,
    IntegrityLabel,
    ResearchRequest,
    Source,
    SourceKind,
)


def _source() -> Source:
    content = (
        "Explicit supervisor routing makes agent delegation inspectable and gives each "
        "quality gate a stable place in the workflow."
    )
    return Source(
        id="S1",
        kind=SourceKind.DEMO,
        title="Inspectable supervisor workflows",
        url="demo://supervisor-workflows",
        content=content,
        provider="bundled-demo",
        integrity=IntegrityLabel.DEMO_FIXTURE,
        checksum=sha256(content.encode()).hexdigest(),
    )


class Pipeline:
    def gather(self, request, queries, uploaded_sources=()):
        del request, queries, uploaded_sources
        return [_source()], []


def _state(max_revisions: int = 1) -> dict[str, object]:
    request = ResearchRequest(
        topic="How should multi-agent research systems be evaluated?",
        max_revisions=max_revisions,
    )
    return {
        "run_id": "run-graph",
        "thread_id": "thread-graph",
        "request": request.model_dump(mode="json"),
        "uploaded_sources": [],
        "sources": [],
        "query_plan": None,
        "synthesis": None,
        "critique": None,
        "report": None,
        "route": "",
        "revision_count": 0,
        "trace": [],
        "warnings": [],
        "conversation": [],
        "error": None,
        "provenance_mode": "demo",
    }


def test_graph_runs_explicit_supervisor_worker_order() -> None:
    graph = build_research_graph(DeterministicResearchModel(), Pipeline())

    final_state = graph.invoke(_state())

    assert final_state["report"] is not None
    assert [event["agent"] for event in final_state["trace"]] == [
        "researcher",
        "summarizer",
        "critic",
        "writer",
    ]
    assert "[S1]" in final_state["report"]["markdown"]


def test_graph_accepts_max_length_topic_and_bounds_final_report_title() -> None:
    graph = build_research_graph(DeterministicResearchModel(), Pipeline())
    state = _state()
    state["request"] = ResearchRequest(topic="T" * 500).model_dump(mode="json")

    final_state = graph.invoke(state)

    assert final_state["report"] is not None
    assert len(final_state["report"]["title"]) == 500


class OneRevisionModel(DeterministicResearchModel):
    def __init__(self) -> None:
        self.review_count = 0

    def critique(self, request, sources, synthesis):
        approved = super().critique(request, sources, synthesis)
        self.review_count += 1
        if self.review_count == 1:
            return Critique(
                disposition=CritiqueDisposition.REVISE_RESEARCH,
                overall_score=0.6,
                citation_coverage=1.0,
                source_quality=0.7,
                gaps=("Compare one additional evidence path.",),
                requested_queries=("multi-agent evaluation comparative evidence",),
            )
        return approved


def test_graph_permits_only_bounded_critic_revision() -> None:
    graph = build_research_graph(OneRevisionModel(), Pipeline())

    final_state = graph.invoke(_state(max_revisions=1))

    assert final_state["revision_count"] == 1
    assert [event["agent"] for event in final_state["trace"]].count("critic") == 2
    assert [event["agent"] for event in final_state["trace"]][-1] == "writer"


def test_graph_exposes_mermaid_topology() -> None:
    mermaid = graph_mermaid(build_research_graph(DeterministicResearchModel(), Pipeline()))

    assert "supervisor" in mermaid
    assert "researcher" in mermaid
    assert "writer" in mermaid
