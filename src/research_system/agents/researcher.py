"""Researcher node: plans queries and acquires a bounded evidence set."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from research_system.agents.common import (
    accepts_conversation_context,
    bounded_prior_conversation,
    completed_trace,
    parse_sources,
)
from research_system.exceptions import ProviderError
from research_system.llm import ResearchModel
from research_system.models import Critique, ResearchRequest, Source
from research_system.state import ResearchState
from research_system.tools.base import deduplicate_sources


class SourceGatherer(Protocol):
    def gather(
        self,
        request: ResearchRequest,
        queries: Sequence[str],
        uploaded_sources: Sequence[Source] = (),
    ) -> tuple[list[Source], list[str]]: ...


def create_researcher_node(
    model: ResearchModel, pipeline: SourceGatherer
) -> Callable[[ResearchState], dict[str, Any]]:
    """Create a dependency-injected LangGraph researcher node."""

    def researcher(state: ResearchState) -> dict[str, Any]:
        request = ResearchRequest.model_validate(state["request"])
        critique = Critique.model_validate(state["critique"]) if state.get("critique") else None
        conversation_context = bounded_prior_conversation(state)
        if conversation_context and accepts_conversation_context(model.plan_queries):
            query_plan = model.plan_queries(  # type: ignore[call-arg]
                request,
                critique,
                conversation_context=conversation_context,
            )
        else:
            query_plan = model.plan_queries(request, critique)
        uploaded = parse_sources(state.get("uploaded_sources"))
        revision_count = int(state.get("revision_count", 0))
        revising = critique is not None and critique.disposition.value == "revise_research"
        prior_sources: list[Source] = []
        if revising:
            revision_count += 1
            prior_sources = parse_sources(state.get("sources"))

        sources, warnings = pipeline.gather(request, query_plan.queries, uploaded)
        if revising:
            # The refreshed evidence set is deliberately ordered before prior search
            # results. Genuine uploads stay first because the pipeline preserves them.
            # Prior evidence then fills any unused budget instead of blocking the
            # critic-requested evidence from entering a full source set.
            sources = deduplicate_sources([*sources, *prior_sources])[: request.max_sources]
        if not sources:
            raise ProviderError("No usable evidence was acquired for this request")
        current_warnings = [str(item) for item in state.get("warnings", [])]
        return {
            "query_plan": query_plan.model_dump(mode="json"),
            "sources": [source.model_dump(mode="json") for source in sources],
            "synthesis": None,
            "critique": None,
            "report": None,
            "revision_count": revision_count,
            "warnings": list(dict.fromkeys([*current_warnings, *warnings])),
            "trace": completed_trace(
                state,
                "researcher",
                f"Acquired {len(sources)} source{'s' if len(sources) != 1 else ''}.",
                {"source_count": len(sources), "query_count": len(query_plan.queries)},
            ),
        }

    return researcher
