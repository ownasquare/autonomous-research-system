"""JSON-serializable LangGraph state contract."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

MAX_CONVERSATION_TURNS = 8
MAX_CONVERSATION_CHARS = 32_000
MAX_STORED_TURN_CHARS = 12_000


def merge_bounded_conversation(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Append checkpointed turns while retaining only a bounded short-term window."""

    combined = [*existing, *incoming]
    selected: list[dict[str, Any]] = []
    remaining = MAX_CONVERSATION_CHARS
    for raw_turn in reversed(combined[-MAX_CONVERSATION_TURNS:]):
        if remaining <= 0:
            break
        content = str(raw_turn.get("content", ""))[:MAX_STORED_TURN_CHARS]
        content = content[:remaining].rstrip()
        if not content:
            continue
        turn = dict(raw_turn)
        turn["content"] = content
        selected.append(turn)
        remaining -= len(content)
    return list(reversed(selected))


class ResearchState(TypedDict, total=False):
    run_id: str
    thread_id: str
    request: dict[str, Any]
    uploaded_sources: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    query_plan: dict[str, Any] | None
    synthesis: dict[str, Any] | None
    critique: dict[str, Any] | None
    report: dict[str, Any] | None
    route: str
    revision_count: int
    trace: list[dict[str, Any]]
    warnings: list[str]
    conversation: Annotated[list[dict[str, Any]], merge_bounded_conversation]
    error: str | None
    provenance_mode: str
