"""Official arXiv export API adapter."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from threading import RLock

import feedparser
import httpx
from pydantic import ValidationError

from research_system.config import Settings
from research_system.exceptions import ProviderError
from research_system.models import IntegrityLabel, Source, SourceKind
from research_system.tools.base import text_checksum

_ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
_RETRYABLE_STATUS_CODES = {408, 425, 429}
_USER_AGENT = "ResearchDesk/0.1 (source-grounded research application)"


class ArxivSearch:
    """Retrieve scholarly records while respecting arXiv rate guidance."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_attempts: int = 3,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._sleep = sleep
        self._clock = clock
        self._max_attempts = max(1, min(max_attempts, 5))
        self._last_request_at: float | None = None
        self._cache: dict[tuple[str, int], list[Source]] = {}
        self._lock = RLock()

    def search(self, query: str, limit: int | None = None) -> list[Source]:
        """Search the official API and normalize Atom entries."""

        with self._lock:
            return self._search_locked(query, limit)

    def _search_locked(self, query: str, limit: int | None) -> list[Source]:
        """Keep cache misses and paced requests serialized per adapter instance."""

        bounded_limit = min(
            limit or self._settings.max_search_results,
            self._settings.max_search_results,
            20,
        )
        cache_key = (query.casefold().strip(), bounded_limit)
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        response = self._get_with_retry(query, bounded_limit)
        if len(response.content) > 5_000_000:
            raise ProviderError("arXiv returned an oversized Atom response")
        feed = feedparser.parse(response.content)
        if feed.bozo and not feed.entries:
            raise ProviderError("arXiv returned invalid Atom XML")

        sources: list[Source] = []
        for entry in feed.entries[:bounded_limit]:
            title = _normalized_text(entry.get("title")) or "Untitled arXiv paper"
            summary = _normalized_text(entry.get("summary"))
            url = _normalized_text(entry.get("id") or entry.get("link"))
            if not url.startswith(("https://", "http://")):
                continue
            authors = tuple(
                name
                for author in entry.get("authors", [])
                if (name := _normalized_text(author.get("name")))
            )
            published_at = _parse_timestamp(entry.get("published"))
            try:
                source = Source(
                    id=f"S{len(sources) + 1}",
                    kind=SourceKind.ARXIV,
                    title=title[:500],
                    url=url,
                    snippet=summary[:20_000],
                    content=summary[:500_000],
                    authors=authors,
                    published_at=published_at,
                    provider="arXiv",
                    integrity=IntegrityLabel.LIVE_ARXIV,
                    checksum=text_checksum(summary or title or url),
                )
            except ValidationError:
                # Atom entries are untrusted provider input. Keep processing
                # when one entry fails the Source security boundary.
                continue
            sources.append(source)
        self._cache[cache_key] = list(sources)
        return sources

    def _get_with_retry(self, query: str, limit: int) -> httpx.Response:
        for attempt in range(self._max_attempts):
            self._respect_rate_limit()
            try:
                with httpx.Client(
                    timeout=self._settings.request_timeout_seconds,
                    transport=self._transport,
                    headers={"User-Agent": _USER_AGENT},
                ) as client:
                    response = client.get(
                        _ARXIV_ENDPOINT,
                        params={
                            "search_query": f"all:{query}",
                            "start": 0,
                            "max_results": limit,
                            "sortBy": "relevance",
                            "sortOrder": "descending",
                        },
                    )
                self._last_request_at = self._clock()
                retryable = (
                    response.status_code in _RETRYABLE_STATUS_CODES or response.status_code >= 500
                )
                if retryable and attempt + 1 < self._max_attempts:
                    continue
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                self._last_request_at = self._clock()
                if attempt + 1 >= self._max_attempts:
                    raise ProviderError("arXiv request failed after bounded retries") from exc
            except httpx.HTTPStatusError as exc:
                raise ProviderError(f"arXiv returned HTTP {exc.response.status_code}") from exc
        raise ProviderError("arXiv request failed after bounded retries")

    def _respect_rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = self._clock() - self._last_request_at
        remaining = self._settings.arxiv_min_interval_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)


def _normalized_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
