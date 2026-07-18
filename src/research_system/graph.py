"""Explicit LangGraph supervisor/worker topology."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_system.agents import (
    create_critic_node,
    create_researcher_node,
    create_summarizer_node,
    create_writer_node,
    supervisor_node,
)
from research_system.agents.researcher import SourceGatherer
from research_system.llm import ResearchModel, build_research_model
from research_system.state import ResearchState

ResearchGraph = CompiledStateGraph[ResearchState, Any, ResearchState, ResearchState]


def _route_from_state(state: ResearchState) -> str:
    return state["route"]


def build_research_graph(
    model: ResearchModel,
    pipeline: SourceGatherer,
    checkpointer: Any | None = None,
) -> ResearchGraph:
    """Compile the custom supervisor loop without prebuilt agent helpers."""

    builder: StateGraph[ResearchState, Any, ResearchState, ResearchState] = StateGraph(
        ResearchState
    )
    builder.add_node("supervisor", supervisor_node, input_schema=ResearchState)
    builder.add_node(
        "researcher",
        RunnableLambda(create_researcher_node(model, pipeline)),
        input_schema=ResearchState,
    )
    builder.add_node(
        "summarizer", RunnableLambda(create_summarizer_node(model)), input_schema=ResearchState
    )
    builder.add_node(
        "critic", RunnableLambda(create_critic_node(model)), input_schema=ResearchState
    )
    builder.add_node(
        "writer", RunnableLambda(create_writer_node(model)), input_schema=ResearchState
    )

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_from_state,
        {
            "researcher": "researcher",
            "summarizer": "summarizer",
            "critic": "critic",
            "writer": "writer",
            "end": END,
        },
    )
    for worker in ("researcher", "summarizer", "critic", "writer"):
        builder.add_edge(worker, "supervisor")
    return builder.compile(checkpointer=checkpointer, name="research-desk")


def graph_mermaid(graph: ResearchGraph) -> str:
    """Return a portable Mermaid rendering of the actual compiled topology."""

    return graph.get_graph().draw_mermaid()


def create_default_graph() -> ResearchGraph:
    """Factory used by LangGraph tooling without opening the run repository."""

    from research_system.config import get_settings
    from research_system.memory.vector import VectorMemory
    from research_system.tools.sources import SourcePipeline

    settings = get_settings()
    memory = VectorMemory(settings.vector_path)
    pipeline = SourcePipeline(settings, memory=memory)
    return build_research_graph(build_research_model(settings), pipeline)
