"""Bundled, explicitly labeled demo evidence."""

from __future__ import annotations

import json
import re
from datetime import datetime
from importlib.resources import files
from typing import Any

from research_system.models import IntegrityLabel, Source, SourceKind
from research_system.tools.base import text_checksum

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
_AGENT_SYSTEM_TERMS = {
    "orchestration",
    "supervisor",
    "critic",
}
_RESEARCH_SYSTEM_TERMS = {
    "agent",
    "agents",
    "agentic",
    "orchestration",
    "supervisor",
    "citation",
    "citations",
    "critic",
    "memory",
    "provenance",
    "synthesis",
    "workflow",
    "workflows",
    "quality",
    "evaluate",
    "evaluated",
    "evaluation",
}
_MULTI_AGENT_PATTERN = re.compile(r"\b(?:multi[- ]agent|multiagent|agentic)\b")


class DemoSearch:
    """Read the package fixture without making a provider request."""

    @staticmethod
    def supports(topic: str) -> bool:
        """Return whether the bundled corpus can truthfully address a topic."""

        normalized = topic.casefold()
        tokens = set(_TOKEN_PATTERN.findall(normalized))
        if _MULTI_AGENT_PATTERN.search(normalized):
            return True
        if tokens & {"agent", "agents"} and tokens & _AGENT_SYSTEM_TERMS:
            return True
        return "research" in tokens and bool(tokens & _RESEARCH_SYSTEM_TERMS)

    def search(self, query: str, limit: int = 8) -> list[Source]:
        raw = files("research_system").joinpath("data/demo_sources.json").read_text("utf-8")
        records: list[dict[str, Any]] = json.loads(raw)
        query_tokens = {token.casefold() for token in query.split() if token}
        ranked = sorted(
            records,
            key=lambda record: _overlap_score(query_tokens, record),
            reverse=True,
        )
        sources: list[Source] = []
        for record in ranked[: max(1, min(limit, 20))]:
            content = str(record["content"])
            sources.append(
                Source(
                    id=f"S{len(sources) + 1}",
                    kind=SourceKind.DEMO,
                    title=str(record["title"]),
                    url=str(record["url"]),
                    snippet=str(record.get("snippet", content[:500])),
                    content=content,
                    authors=tuple(str(author) for author in record.get("authors", [])),
                    published_at=datetime.fromisoformat(str(record["published_at"])),
                    provider="bundled demo corpus",
                    integrity=IntegrityLabel.DEMO_FIXTURE,
                    locator=str(record.get("locator") or "Bundled snapshot"),
                    checksum=text_checksum(content),
                )
            )
        return sources


def _overlap_score(query_tokens: set[str], record: dict[str, Any]) -> int:
    haystack = f"{record.get('title', '')} {record.get('content', '')}".casefold()
    return sum(token in haystack for token in query_tokens)
