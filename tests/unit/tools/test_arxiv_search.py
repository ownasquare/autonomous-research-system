from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import httpx
import pytest

from research_system.config import Settings
from research_system.exceptions import ProviderError
from research_system.models import IntegrityLabel, Source, SourceKind
from research_system.tools.arxiv import ArxivSearch

_ATOM_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2607.00001</id>
    <published>2026-07-01T00:00:00Z</published>
    <title>Supervisor graphs for research agents</title>
    <summary>Explicit state improves auditability and bounded revision.</summary>
    <author><name>Ada Researcher</name></author>
  </entry>
</feed>
"""


def _atom_response_with_urls(*urls: str) -> str:
    entries = "".join(
        f"""
  <entry>
    <id>{url}</id>
    <published>2026-07-01T00:00:00Z</published>
    <title>Paper {index}</title>
    <summary>Source-grounded evidence {index}.</summary>
  </entry>
"""
        for index, url in enumerate(urls, start=1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>
"""


def test_arxiv_retries_with_pacing_parses_atom_and_caches(tmp_path) -> None:
    requests: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["user-agent"].startswith("ResearchDesk/")
        if len(requests) == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, text=_ATOM_RESPONSE, request=request)

    adapter = ArxivSearch(
        Settings(research_data_dir=tmp_path, arxiv_min_interval_seconds=3),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        clock=lambda: 100.0,
    )

    first = adapter.search("agent graphs", limit=2)
    second = adapter.search("agent graphs", limit=2)

    assert len(requests) == 2
    assert sleeps == [3.0]
    assert first == second
    assert first[0].kind is SourceKind.ARXIV
    assert first[0].integrity is IntegrityLabel.LIVE_ARXIV
    assert first[0].authors == ("Ada Researcher",)


def test_arxiv_non_retryable_status_is_typed(tmp_path) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(400, request=request)

    adapter = ArxivSearch(
        Settings(research_data_dir=tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    with pytest.raises(ProviderError, match="HTTP 400"):
        adapter.search("bad query")

    assert request_count == 1


def test_arxiv_never_exceeds_configured_result_limit(tmp_path) -> None:
    requested_limits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_limits.append(request.url.params["max_results"])
        return httpx.Response(200, text=_ATOM_RESPONSE, request=request)

    adapter = ArxivSearch(
        Settings(research_data_dir=tmp_path, max_search_results=2),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    adapter.search("agent graphs", limit=20)

    assert requested_limits == ["2"]


def test_arxiv_serializes_concurrent_cache_misses_for_required_pacing(tmp_path) -> None:
    requests: list[httpx.Request] = []
    sleeps: list[float] = []
    requests_lock = Lock()
    start = Barrier(2)

    def handler(request: httpx.Request) -> httpx.Response:
        with requests_lock:
            requests.append(request)
        time.sleep(0.05)
        return httpx.Response(200, text=_ATOM_RESPONSE, request=request)

    adapter = ArxivSearch(
        Settings(research_data_dir=tmp_path, arxiv_min_interval_seconds=3),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        clock=lambda: 100.0,
    )

    def run_search(query: str) -> list[Source]:
        start.wait()
        return adapter.search(query, limit=1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_search, "agent graphs")
        second = executor.submit(run_search, "research agents")
        assert first.result()
        assert second.result()

    assert len(requests) == 2
    assert sleeps == [3.0]


@pytest.mark.parametrize(
    "malicious_url",
    [
        "https://arxiv.org/abs/2607.00001 bad",
        "https://arxiv.org/abs/2607.00001\tfragment",
        "https://arxiv.org/abs/2607.00001\nhttps://evil.test/pixel",
        "https://user:password@arxiv.org/abs/2607.00001",
    ],
    ids=["whitespace", "control", "newline", "userinfo"],
)
def test_arxiv_skips_invalid_url_and_preserves_valid_sibling(tmp_path, malicious_url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_atom_response_with_urls(
                malicious_url,
                "https://arxiv.org/abs/2607.00002",
            ),
            request=request,
        )

    adapter = ArxivSearch(
        Settings(research_data_dir=tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    sources = adapter.search("agent graphs", limit=2)

    assert [(source.id, source.title, source.url) for source in sources] == [
        ("S1", "Paper 2", "https://arxiv.org/abs/2607.00002")
    ]
