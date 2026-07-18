from __future__ import annotations

from hashlib import sha256

import pytest

from research_system.agents.critic import create_critic_node
from research_system.agents.researcher import create_researcher_node
from research_system.agents.summarizer import create_summarizer_node
from research_system.agents.writer import create_writer_node
from research_system.exceptions import CitationIntegrityError
from research_system.llm import DeterministicResearchModel, ReportDraft
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
        "Independent critic stages improve research quality by identifying evidence gaps "
        "before the final report is written."
    )
    return Source(
        id="S1",
        kind=SourceKind.DEMO,
        title="Critic stages and quality",
        url="demo://critic-quality",
        content=content,
        provider="bundled-demo",
        integrity=IntegrityLabel.DEMO_FIXTURE,
        checksum=sha256(content.encode()).hexdigest(),
    )


class StubPipeline:
    def gather(
        self,
        request: ResearchRequest,
        queries: tuple[str, ...],
        uploaded_sources: tuple[Source, ...] | list[Source] = (),
    ) -> tuple[list[Source], list[str]]:
        del request, queries, uploaded_sources
        return [_source()], []


def _initial_state() -> dict[str, object]:
    request = ResearchRequest(topic="How should multi-agent research systems be evaluated?")
    return {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "request": request.model_dump(mode="json"),
        "uploaded_sources": [],
        "sources": [],
        "trace": [],
        "warnings": [],
        "revision_count": 0,
        "provenance_mode": "demo",
    }


def test_workers_build_a_complete_cited_report() -> None:
    model = DeterministicResearchModel()
    state = _initial_state()
    state.update(create_researcher_node(model, StubPipeline())(state))
    state.update(create_summarizer_node(model)(state))
    state.update(create_critic_node(model)(state))
    state.update(create_writer_node(model)(state))

    assert state["report"] is not None
    report = state["report"]
    assert isinstance(report, dict)
    assert "[S1]" in str(report["markdown"])
    assert [event["agent"] for event in state["trace"]] == [
        "researcher",
        "summarizer",
        "critic",
        "writer",
    ]


class HallucinatingWriterModel(DeterministicResearchModel):
    def write_report(self, *args: object, **kwargs: object) -> ReportDraft:
        del args, kwargs
        return ReportDraft(
            title="A report with an invalid citation",
            executive_summary="This otherwise plausible summary cites evidence that is absent.",
            markdown=(
                "# Invalid citation report\n\nThis claim points to evidence that was never "
                "retrieved [S9].\n\n## Conclusion\n\nThe citation must be rejected."
            ),
            source_ids=("S9",),
            limitations=(),
        )


def test_writer_blocks_unknown_source_ids() -> None:
    model = HallucinatingWriterModel()
    state = _initial_state()
    state.update(create_researcher_node(model, StubPipeline())(state))
    state.update(create_summarizer_node(model)(state))
    state.update(create_critic_node(model)(state))

    with pytest.raises(CitationIntegrityError, match="S9"):
        create_writer_node(model)(state)


def test_summary_revision_receives_and_addresses_critic_feedback() -> None:
    state = _initial_state()
    state["sources"] = [_source().model_dump(mode="json")]
    state["critique"] = Critique(
        disposition=CritiqueDisposition.REVISE_SUMMARY,
        overall_score=0.6,
        citation_coverage=1.0,
        source_quality=0.8,
        gaps=("Explain which quality control should guide adoption.",),
    ).model_dump(mode="json")

    result = create_summarizer_node(DeterministicResearchModel())(state)

    synthesis = result["synthesis"]
    assert (
        "which quality control should guide adoption" in synthesis["executive_summary"].casefold()
    )
    assert result["revision_count"] == 1


def test_research_revision_prioritizes_upload_then_new_evidence_before_prior_set() -> None:
    def source(
        source_id: str,
        label: str,
        *,
        kind: SourceKind = SourceKind.WEB,
        integrity: IntegrityLabel = IntegrityLabel.LIVE_WEB,
    ) -> Source:
        content = f"Evidence for {label}"
        scheme = "upload" if kind is SourceKind.PDF else "https"
        return Source(
            id=source_id,
            kind=kind,
            title=label,
            url=f"{scheme}://evidence.test/{label}",
            content=content,
            provider="revision test",
            integrity=integrity,
            checksum=sha256(content.encode()).hexdigest(),
        )

    uploaded = source(
        "S1",
        "uploaded",
        kind=SourceKind.PDF,
        integrity=IntegrityLabel.USER_UPLOAD,
    )
    prior_one = source("S2", "prior-one")
    prior_two = source("S3", "prior-two")
    new_evidence = source("S8", "new-critic-evidence")

    class RevisionPipeline:
        def __init__(self) -> None:
            self.received_uploads: list[Source] = []

        def gather(
            self,
            request: ResearchRequest,
            queries: tuple[str, ...],
            uploaded_sources: tuple[Source, ...] | list[Source] = (),
        ) -> tuple[list[Source], list[str]]:
            del request
            assert queries == ("new critic evidence",)
            self.received_uploads = list(uploaded_sources)
            return [*uploaded_sources, new_evidence], []

    pipeline = RevisionPipeline()
    state = _initial_state()
    state["request"] = ResearchRequest(
        topic="How should multi-agent research systems be evaluated?",
        max_sources=3,
    ).model_dump(mode="json")
    state["uploaded_sources"] = [uploaded.model_dump(mode="json")]
    state["sources"] = [
        uploaded.model_dump(mode="json"),
        prior_one.model_dump(mode="json"),
        prior_two.model_dump(mode="json"),
    ]
    state["critique"] = Critique(
        disposition=CritiqueDisposition.REVISE_RESEARCH,
        overall_score=0.6,
        citation_coverage=0.7,
        source_quality=0.5,
        requested_queries=("new critic evidence",),
    ).model_dump(mode="json")

    result = create_researcher_node(DeterministicResearchModel(), pipeline)(state)
    revised = [Source.model_validate(item) for item in result["sources"]]

    assert [item.title for item in pipeline.received_uploads] == ["uploaded"]
    assert [item.title for item in revised] == [
        "uploaded",
        "new-critic-evidence",
        "prior-one",
    ]
    assert [item.id for item in revised] == ["S1", "S2", "S3"]
    assert result["revision_count"] == 1
