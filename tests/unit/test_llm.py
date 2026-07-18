from __future__ import annotations

import json
from hashlib import sha256

import pytest

from research_system.config import Settings
from research_system.exceptions import CitationIntegrityError
from research_system.llm import (
    MODEL_SOURCE_EVIDENCE_CHAR_BUDGET,
    MODEL_TOTAL_EVIDENCE_CHAR_BUDGET,
    DeterministicResearchModel,
    OpenAIResearchModel,
    ReportDraft,
    validate_citations,
)
from research_system.models import (
    Critique,
    CritiqueDisposition,
    Finding,
    IntegrityLabel,
    QueryPlan,
    ResearchDepth,
    ResearchRequest,
    Source,
    SourceKind,
    Synthesis,
)


def _source(source_id: str = "S1") -> Source:
    content = (
        "Structured agent workflows make delegation observable and enable independent "
        "quality checks before a report is accepted."
    )
    return Source(
        id=source_id,
        kind=SourceKind.DEMO,
        title="Observable agent workflows",
        url=f"demo://{source_id.lower()}",
        content=content,
        provider="bundled-demo",
        integrity=IntegrityLabel.DEMO_FIXTURE,
        checksum=sha256(content.encode()).hexdigest(),
    )


def test_deterministic_model_produces_valid_structured_artifacts() -> None:
    model = DeterministicResearchModel()
    request = ResearchRequest(topic="How should multi-agent research systems be evaluated?")
    sources = [_source()]

    plan = model.plan_queries(request)
    synthesis = model.synthesize(request, sources)
    critique = model.critique(request, sources, synthesis)
    draft = model.write_report(request, sources, synthesis, critique)

    assert request.topic in plan.queries[0]
    assert synthesis.findings[0].source_ids == ("S1",)
    assert "?." not in synthesis.executive_summary
    assert critique.disposition.value == "approved"
    assert "[S1]" in draft.markdown


def test_deterministic_model_bounds_title_for_max_length_topic() -> None:
    model = DeterministicResearchModel()
    request = ResearchRequest(topic="T" * 500)
    sources = [_source()]

    synthesis = model.synthesize(request, sources)
    critique = model.critique(request, sources, synthesis)
    draft = model.write_report(request, sources, synthesis, critique)

    assert len(draft.title) == 500
    assert draft.title.endswith("…")


def test_deterministic_follow_up_answers_question_with_cited_evidence() -> None:
    model = DeterministicResearchModel()
    question = "Which quality control should guide adoption?"
    request = ResearchRequest(
        topic="How should multi-agent research systems be evaluated?",
        follow_up=question,
    )
    sources = [_source()]

    synthesis = model.synthesize(request, sources)
    critique = model.critique(request, sources, synthesis)
    draft = model.write_report(request, sources, synthesis, critique)

    assert "## Follow-up answer" in draft.markdown
    assert question in draft.markdown
    assert "Structured agent workflows make delegation observable" in draft.markdown
    assert "[S1]" in draft.markdown


def test_deterministic_comparison_follow_up_renders_cited_table() -> None:
    model = DeterministicResearchModel()
    request = ResearchRequest(
        topic="How should multi-agent research systems be evaluated?",
        follow_up="Compare the strongest approaches in a table.",
    )
    sources = [_source()]

    synthesis = model.synthesize(request, sources)
    critique = model.critique(request, sources, synthesis)
    draft = model.write_report(request, sources, synthesis, critique)

    assert "| Evidence perspective | Finding | Sources |" in draft.markdown
    assert "| Perspective 1 |" in draft.markdown
    assert "[S1]" in draft.markdown


class _FakeStructuredRunnable:
    def __init__(self, client: _FakeStructuredClient, schema: type[object]) -> None:
        self.client = client
        self.schema = schema

    def invoke(self, messages: list[tuple[str, str]]) -> object:
        payload = json.loads(messages[1][1])
        self.client.calls.append((self.schema, payload))
        if self.schema is QueryPlan:
            return QueryPlan(
                queries=("bounded evidence orchestration",),
                rationale="The query targets the requested research boundary.",
            )
        if self.schema is Synthesis:
            return Synthesis(
                executive_summary=(
                    "The bounded prompt retains enough evidence to produce a grounded summary."
                ),
                themes=("Evidence",),
                findings=(
                    Finding(
                        claim="The first source supports explicit evidence budgeting.",
                        source_ids=("S1",),
                        confidence=0.8,
                    ),
                ),
            )
        if self.schema is Critique:
            return Critique(
                disposition=CritiqueDisposition.APPROVED,
                overall_score=0.9,
                citation_coverage=1.0,
                source_quality=0.8,
            )
        if self.schema is ReportDraft:
            return ReportDraft(
                title="Bounded hosted-model evidence",
                executive_summary=(
                    "The hosted-model adapter receives bounded evidence and stable metadata."
                ),
                markdown=(
                    "# Bounded hosted-model evidence\n\n"
                    "The synthesis stays grounded in the first retained source [S1].\n\n"
                    "## Conclusion\n\nExplicit prompt budgets make worst-case requests predictable."
                ),
                source_ids=("S1",),
            )
        raise AssertionError(f"Unexpected schema: {self.schema}")


class _FakeStructuredClient:
    def __init__(self) -> None:
        self.calls: list[tuple[type[object], dict[str, object]]] = []

    def with_structured_output(self, schema: type[object]) -> _FakeStructuredRunnable:
        return _FakeStructuredRunnable(self, schema)


def test_openai_model_calls_are_structured_and_worst_case_evidence_is_bounded(tmp_path) -> None:
    content = "C" * 500_000
    snippet = "S" * 20_000
    sources = [
        Source(
            id=f"S{index}",
            kind=SourceKind.WEB,
            title=f"Source {index} " + ("T" * (500 - len(f"Source {index} "))),
            url="https://example.test/" + ("u" * (2_000 - len("https://example.test/"))),
            snippet=snippet,
            content=content,
            provider="p" * 100,
            integrity=IntegrityLabel.LIVE_WEB,
            locator="l" * 500,
            checksum="a" * 64,
        )
        for index in range(1, 21)
    ]
    request = ResearchRequest(
        topic="How should hosted research models budget evidence?",
        depth=ResearchDepth.DEEP,
    )
    client = _FakeStructuredClient()
    model = OpenAIResearchModel(
        Settings(research_data_dir=tmp_path),
        client=client,  # type: ignore[arg-type]
    )

    plan = model.plan_queries(request)
    synthesis = model.synthesize(request, sources)
    critique = model.critique(request, sources, synthesis)
    draft = model.write_report(request, sources, synthesis, critique)

    assert plan.queries
    assert draft.source_ids == ("S1",)
    assert [schema for schema, _ in client.calls] == [
        QueryPlan,
        Synthesis,
        Critique,
        ReportDraft,
    ]
    synthesis_payload = next(payload for schema, payload in client.calls if schema is Synthesis)
    prompt_sources = synthesis_payload["sources"]
    assert isinstance(prompt_sources, list)
    assert len(prompt_sources) == 20
    required_metadata = {"id", "title", "url", "provider", "integrity", "locator"}
    assert all(required_metadata <= record.keys() for record in prompt_sources)
    evidence_lengths = [
        len(str(record["snippet_excerpt"])) + len(str(record["content_excerpt"]))
        for record in prompt_sources
    ]
    assert max(evidence_lengths) <= MODEL_SOURCE_EVIDENCE_CHAR_BUDGET
    assert sum(evidence_lengths) <= MODEL_TOTAL_EVIDENCE_CHAR_BUDGET
    assert all(length > 0 for length in evidence_lengths)
    assert all(len(json.dumps(payload)) < 150_000 for _, payload in client.calls)


def test_citation_validation_blocks_unknown_ids() -> None:
    with pytest.raises(CitationIntegrityError, match="S9"):
        validate_citations("A claim cites an unavailable source [S9].", {"S1"})


def test_citation_validation_requires_at_least_one_known_citation() -> None:
    with pytest.raises(CitationIntegrityError, match="at least one"):
        validate_citations("A report with no evidence marker.", {"S1"})
