"""Deterministic supervisor routing for the explicit StateGraph loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from research_system.state import ResearchState

Route = Literal["researcher", "summarizer", "critic", "writer", "end"]


def choose_route(state: Mapping[str, Any]) -> Route:
    """Choose the next specialist while enforcing the request revision budget."""

    if state.get("error") or state.get("report"):
        return "end"
    if not state.get("sources"):
        return "researcher"
    if not state.get("synthesis"):
        return "summarizer"
    if not state.get("critique"):
        return "critic"

    critique = state["critique"]
    disposition = str(critique.get("disposition", "approved"))
    request = state.get("request", {})
    revision_count = int(state.get("revision_count", 0))
    max_revisions = int(request.get("max_revisions", 0))
    if revision_count < max_revisions:
        if disposition == "revise_research":
            return "researcher"
        if disposition == "revise_summary":
            return "summarizer"
    return "writer"


def supervisor_node(state: ResearchState) -> dict[str, Any]:
    """LangGraph node that makes routing state visible and serializable."""

    route = choose_route(state)
    update: dict[str, Any] = {"route": route}
    critique = state.get("critique")
    if route == "writer" and isinstance(critique, Mapping):
        disposition = str(critique.get("disposition", "approved"))
        request = state.get("request", {})
        revision_count = int(state.get("revision_count", 0))
        max_revisions = int(request.get("max_revisions", 0))
        if disposition != "approved" and revision_count >= max_revisions:
            warning = (
                "The critic requested another revision, but the configured revision "
                f"budget ({max_revisions}) was exhausted. The report was finalized with "
                "unresolved review feedback."
            )
            update["warnings"] = list(
                dict.fromkeys([*(str(item) for item in state.get("warnings", [])), warning])
            )
    return update
