"""Tavily search adapter with bounded retries and explicit HTTP handling."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import ValidationError

from research_system.config import Settings
from research_system.exceptions import ProviderError
from research_system.models import IntegrityLabel, Source, SourceKind
from research_system.tools.base import text_checksum

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_RETRYABLE_STATUS_CODES = {408, 425, 429}


class TavilySearch:
    """Retrieve normalized live-web evidence from Tavily."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 3,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._sleep = sleep
        self._max_attempts = max(1, min(max_attempts, 5))

    def search(self, query: str, limit: int | None = None) -> list[Source]:
        """Search Tavily, rejecting unconfigured use before any request."""

        if self._settings.tavily_api_key is None:
            raise ProviderError("Tavily is not configured")
        bounded_limit = min(
            limit or self._settings.max_search_results,
            self._settings.max_search_results,
            20,
        )
        payload = {
            "api_key": self._settings.tavily_api_key.get_secret_value(),
            "query": query,
            "max_results": bounded_limit,
            "include_answer": False,
            "include_raw_content": True,
            "search_depth": "advanced",
        }
        response = self._post_with_retry(payload)
        body: dict[str, Any]
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError("Tavily returned invalid JSON") from exc
        raw_results = body.get("results", [])
        if not isinstance(raw_results, list):
            raise ProviderError("Tavily returned an invalid result collection")

        sources: list[Source] = []
        for item in raw_results[:bounded_limit]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Untitled web result")[:500]
            url = str(item.get("url") or "")
            if not url.startswith(("https://", "http://")):
                continue
            snippet = str(item.get("content") or "")[:20_000]
            content = str(item.get("raw_content") or snippet)[:500_000]
            raw_score = item.get("score")
            score = float(raw_score) if isinstance(raw_score, int | float) else None
            if score is not None:
                score = min(max(score, 0.0), 1.0)
            try:
                source = Source(
                    id=f"S{len(sources) + 1}",
                    kind=SourceKind.WEB,
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=content,
                    provider="Tavily",
                    integrity=IntegrityLabel.LIVE_WEB,
                    score=score,
                    checksum=text_checksum(content or snippet or url),
                )
            except ValidationError:
                # Provider result sets are untrusted. Isolate a malformed entry
                # instead of discarding valid sibling evidence.
                continue
            sources.append(source)
        return sources

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(self._max_attempts):
            try:
                with httpx.Client(
                    timeout=self._settings.request_timeout_seconds,
                    transport=self._transport,
                ) as client:
                    response = client.post(_TAVILY_ENDPOINT, json=payload)
                retryable = (
                    response.status_code in _RETRYABLE_STATUS_CODES or response.status_code >= 500
                )
                if retryable and attempt + 1 < self._max_attempts:
                    self._sleep(0.25 * (2**attempt))
                    continue
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt + 1 >= self._max_attempts:
                    raise ProviderError("Tavily request failed after bounded retries") from exc
                self._sleep(0.25 * (2**attempt))
            except httpx.HTTPStatusError as exc:
                raise ProviderError(f"Tavily returned HTTP {exc.response.status_code}") from exc
        raise ProviderError("Tavily request failed after bounded retries")
