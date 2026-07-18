"""Shared source normalization helpers."""

from __future__ import annotations

from hashlib import sha256
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from research_system.models import Source

_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref"}


def text_checksum(value: str) -> str:
    """Return a stable checksum for source or report text."""

    return sha256(value.encode("utf-8")).hexdigest()


def canonical_url(value: str) -> str:
    """Normalize an HTTP URL without changing its evidence destination."""

    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        return value
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_KEYS
    ]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            urlencode(sorted(query)),
            "",
        )
    )


def normalize_source_ids(sources: list[Source]) -> list[Source]:
    """Assign stable citation labels after acquisition and deduplication."""

    return [
        source.model_copy(update={"id": f"S{index}"}) for index, source in enumerate(sources, 1)
    ]


def deduplicate_sources(sources: list[Source]) -> list[Source]:
    """Deduplicate evidence in order and assign contiguous citation labels."""

    unique: list[Source] = []
    seen: set[str] = set()
    for source in sources:
        if source.url.startswith(("https://", "http://")):
            key = f"url:{canonical_url(source.url)}"
        else:
            key = f"content:{source.checksum}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return normalize_source_ids(unique)
