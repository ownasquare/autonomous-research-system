"""Report Writer node with mandatory citation-integrity validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from research_system.agents.common import (
    accepts_conversation_context,
    bounded_prior_conversation,
    completed_trace,
    parse_sources,
)
from research_system.exceptions import CitationIntegrityError
from research_system.llm import ResearchModel, validate_citations
from research_system.models import (
    ConversationTurn,
    Critique,
    ResearchMode,
    ResearchReport,
    ResearchRequest,
    Synthesis,
)
from research_system.state import ResearchState


def _safe_markdown_label(value: str) -> str:
    return " ".join(value.replace("[", "(").replace("]", ")").split())


def calculate_citation_coverage(markdown: str) -> float:
    """Estimate paragraph-level citation coverage for substantive report prose."""

    paragraphs = [
        paragraph.strip()
        for paragraph in markdown.split("\n\n")
        if len(paragraph.split()) >= 8 and not paragraph.lstrip().startswith(("#", "-"))
    ]
    if not paragraphs:
        return 0.0
    cited = sum("[S" in paragraph for paragraph in paragraphs)
    return cited / len(paragraphs)


def create_writer_node(model: ResearchModel) -> Callable[[ResearchState], dict[str, Any]]:
    """Create a dependency-injected LangGraph writer node."""

    def writer(state: ResearchState) -> dict[str, Any]:
        request = ResearchRequest.model_validate(state["request"])
        sources = parse_sources(state.get("sources"))
        synthesis = Synthesis.model_validate(state["synthesis"])
        critique = Critique.model_validate(state["critique"])
        conversation_context = bounded_prior_conversation(state)
        if conversation_context and accepts_conversation_context(model.write_report):
            draft = model.write_report(  # type: ignore[call-arg]
                request,
                sources,
                synthesis,
                critique,
                conversation_context=conversation_context,
            )
        else:
            draft = model.write_report(request, sources, synthesis, critique)

        known_ids = {source.id for source in sources}
        declared_ids = set(draft.source_ids)
        unknown_declared = declared_ids - known_ids
        if unknown_declared:
            raise CitationIntegrityError(
                f"Report declares unknown source IDs: {', '.join(sorted(unknown_declared))}"
            )
        cited_ids = validate_citations(draft.markdown, known_ids)
        if declared_ids != cited_ids:
            raise CitationIntegrityError(
                "Report source inventory must exactly match its inline citations"
            )

        references = ["## Sources"]
        for source in sources:
            if source.id not in cited_ids:
                continue
            locator = f" — {_safe_markdown_label(source.locator)}" if source.locator else ""
            references.append(
                f"- [{source.id}] {_safe_markdown_label(source.title)}{locator}: {source.url}"
            )
        markdown = f"{draft.markdown.rstrip()}\n\n" + "\n".join(references)
        report = ResearchReport(
            run_id=state["run_id"],
            thread_id=state["thread_id"],
            topic=request.topic,
            title=draft.title,
            executive_summary=draft.executive_summary,
            markdown=markdown,
            source_ids=tuple(source.id for source in sources if source.id in cited_ids),
            critique=critique,
            limitations=draft.limitations,
            provenance_mode=ResearchMode(state.get("provenance_mode", "demo")),
        )
        critic_warnings = [
            f"Critic flagged unsupported claim: {claim}" for claim in critique.unsupported_claims
        ]
        warnings = list(
            dict.fromkeys([*[str(item) for item in state.get("warnings", [])], *critic_warnings])
        )
        return {
            "report": report.model_dump(mode="json"),
            "warnings": warnings,
            "conversation": [
                ConversationTurn(role="assistant", content=report.markdown).model_dump(mode="json")
            ],
            "trace": completed_trace(
                state,
                "writer",
                f"Completed a cited report using {len(report.source_ids)} sources.",
                {
                    "source_count": len(report.source_ids),
                    "citation_coverage": calculate_citation_coverage(report.markdown),
                },
            ),
        }

    return writer
