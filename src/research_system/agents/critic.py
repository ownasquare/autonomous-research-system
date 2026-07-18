"""Critic node: evaluates coverage and requests bounded revision when needed."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from research_system.agents.common import completed_trace, parse_sources
from research_system.llm import ResearchModel
from research_system.models import ResearchRequest, Synthesis
from research_system.state import ResearchState


def create_critic_node(model: ResearchModel) -> Callable[[ResearchState], dict[str, Any]]:
    """Create a dependency-injected LangGraph critic node."""

    def critic(state: ResearchState) -> dict[str, Any]:
        request = ResearchRequest.model_validate(state["request"])
        sources = parse_sources(state.get("sources"))
        synthesis = Synthesis.model_validate(state["synthesis"])
        critique = model.critique(request, sources, synthesis)
        return {
            "critique": critique.model_dump(mode="json"),
            "report": None,
            "trace": completed_trace(
                state,
                "critic",
                f"Quality review completed with a {critique.overall_score:.0%} score.",
                {
                    "overall_score": critique.overall_score,
                    "citation_coverage": critique.citation_coverage,
                    "disposition": critique.disposition.value,
                },
            ),
        }

    return critic
