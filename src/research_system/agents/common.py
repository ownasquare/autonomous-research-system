"""Shared JSON-state helpers for focused agent nodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from inspect import Parameter, signature
from typing import Any, Literal, cast

from research_system.models import AgentTraceEvent, ConversationTurn, Source

AgentName = Literal["researcher", "summarizer", "critic", "writer"]
MAX_MODEL_CONTEXT_TURNS = 4
MAX_MODEL_CONTEXT_CHARS = 6_000
MAX_MODEL_TURN_CHARS = 4_000


def parse_sources(raw_sources: Sequence[object] | None) -> list[Source]:
    return [Source.model_validate(source) for source in (raw_sources or [])]


def bounded_prior_conversation(state: Mapping[str, Any]) -> tuple[ConversationTurn, ...]:
    """Return a compact prior-turn window, excluding the current user request."""

    turns = [
        ConversationTurn.model_validate(raw_turn) for raw_turn in state.get("conversation", [])
    ]
    if turns and turns[-1].role == "user":
        turns.pop()

    selected: list[ConversationTurn] = []
    remaining = MAX_MODEL_CONTEXT_CHARS
    for turn in reversed(turns[-MAX_MODEL_CONTEXT_TURNS:]):
        if remaining <= 0:
            break
        content = turn.content[: min(MAX_MODEL_TURN_CHARS, remaining)].rstrip()
        if not content:
            continue
        selected.append(turn.model_copy(update={"content": content}))
        remaining -= len(content)
    return tuple(reversed(selected))


def accepts_conversation_context(method: Callable[..., object]) -> bool:
    """Detect the additive context keyword without breaking legacy model adapters."""

    try:
        parameters = signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "conversation_context" or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def accepts_critic_feedback(method: Callable[..., object]) -> bool:
    """Detect additive revision feedback without breaking legacy model adapters."""

    try:
        parameters = signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "critic_feedback" or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def completed_trace(
    state: Mapping[str, Any],
    agent: AgentName,
    message: str,
    details: dict[str, str | int | float | bool] | None = None,
) -> list[dict[str, Any]]:
    trace = [cast(dict[str, Any], dict(item)) for item in state.get("trace", [])]
    event = AgentTraceEvent(
        sequence=len(trace) + 1,
        agent=agent,
        status="completed",
        message=message,
        details=details or {},
    )
    return [*trace, event.model_dump(mode="json")]
