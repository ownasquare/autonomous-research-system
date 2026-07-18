"""Summarizer node: maps evidence into source-linked findings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from research_system.agents.common import accepts_critic_feedback, completed_trace, parse_sources
from research_system.exceptions import CitationIntegrityError
from research_system.llm import ResearchModel
from research_system.models import Critique, ResearchRequest
from research_system.state import ResearchState


def create_summarizer_node(model: ResearchModel) -> Callable[[ResearchState], dict[str, Any]]:
    """Create a dependency-injected LangGraph summarizer node."""

    def summarizer(state: ResearchState) -> dict[str, Any]:
        request = ResearchRequest.model_validate(state["request"])
        sources = parse_sources(state.get("sources"))
        prior_critique = (
            Critique.model_validate(state["critique"]) if state.get("critique") else None
        )
        revising = (
            prior_critique is not None and prior_critique.disposition.value == "revise_summary"
        )
        if revising and accepts_critic_feedback(model.synthesize):
            synthesis = model.synthesize(
                request,
                sources,
                critic_feedback=prior_critique,
            )
        else:
            synthesis = model.synthesize(request, sources)
        known_ids = {source.id for source in sources}
        used_ids = {source_id for finding in synthesis.findings for source_id in finding.source_ids}
        unknown = used_ids - known_ids
        if unknown:
            raise CitationIntegrityError(
                f"Synthesis contains unknown source IDs: {', '.join(sorted(unknown))}"
            )

        revision_count = int(state.get("revision_count", 0))
        if revising:
            revision_count += 1
        return {
            "synthesis": synthesis.model_dump(mode="json"),
            "critique": None,
            "report": None,
            "revision_count": revision_count,
            "trace": completed_trace(
                state,
                "summarizer",
                f"Organized evidence into {len(synthesis.findings)} grounded findings.",
                {"finding_count": len(synthesis.findings)},
            ),
        }

    return summarizer
