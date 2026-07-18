from __future__ import annotations

import json

import httpx
import pytest

from research_system.config import Settings
from research_system.exceptions import ProviderError
from research_system.models import IntegrityLabel, ResearchMode, SourceKind
from research_system.tools.tavily import TavilySearch


def test_tavily_retries_transient_status_and_normalizes_result(tmp_path) -> None:
    request_count = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.extensions["timeout"]["read"] == 7.0
        if request_count == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Supervising research agents",
                        "url": "https://example.test/agents",
                        "content": "A concise evidence snippet.",
                        "raw_content": "Full source-grounded evidence.",
                        "score": 0.91,
                    }
                ]
            },
            request=request,
        )

    settings = Settings(
        research_mode=ResearchMode.LIVE,
        research_data_dir=tmp_path,
        tavily_api_key="test-provider-key",
        request_timeout_seconds=7,
    )
    adapter = TavilySearch(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    sources = adapter.search("research agents", limit=2)

    assert request_count == 2
    assert sleeps == [0.25]
    assert sources[0].kind is SourceKind.WEB
    assert sources[0].integrity is IntegrityLabel.LIVE_WEB
    assert sources[0].content == "Full source-grounded evidence."


def test_tavily_requires_configuration_before_network(tmp_path) -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"results": []})

    adapter = TavilySearch(
        Settings(research_mode=ResearchMode.LIVE, research_data_dir=tmp_path),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProviderError, match="not configured"):
        adapter.search("research agents")

    assert called is False


def test_tavily_timeout_retries_are_bounded(tmp_path) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        raise httpx.ReadTimeout("test timeout", request=request)

    adapter = TavilySearch(
        Settings(
            research_mode=ResearchMode.LIVE,
            research_data_dir=tmp_path,
            tavily_api_key="test-provider-key",
        ),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    with pytest.raises(ProviderError, match="bounded retries"):
        adapter.search("research agents")

    assert request_count == 3


def test_tavily_never_exceeds_configured_result_limit(tmp_path) -> None:
    requested_limits: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_limits.append(int(json.loads(request.content)["max_results"]))
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": f"Result {index}",
                        "url": f"https://example.test/{index}",
                        "content": f"Evidence {index}",
                    }
                    for index in range(4)
                ]
            },
            request=request,
        )

    adapter = TavilySearch(
        Settings(
            research_mode=ResearchMode.LIVE,
            research_data_dir=tmp_path,
            tavily_api_key="test-provider-key",
            max_search_results=2,
        ),
        transport=httpx.MockTransport(handler),
    )

    sources = adapter.search("research agents", limit=20)

    assert requested_limits == [2]
    assert len(sources) == 2


@pytest.mark.parametrize(
    "malicious_url",
    [
        "https://example.test/report bad",
        "https://example.test/report\tfragment",
        "https://example.test/report\nhttps://evil.test/pixel",
        "https://user:password@example.test/report",
    ],
    ids=["whitespace", "control", "newline", "userinfo"],
)
def test_tavily_skips_invalid_url_and_preserves_valid_sibling(tmp_path, malicious_url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Untrusted result",
                        "url": malicious_url,
                        "content": "This entry must be isolated.",
                    },
                    {
                        "title": "Valid sibling",
                        "url": "https://example.test/valid-report",
                        "content": "This evidence must survive.",
                    },
                ]
            },
            request=request,
        )

    adapter = TavilySearch(
        Settings(
            research_mode=ResearchMode.LIVE,
            research_data_dir=tmp_path,
            tavily_api_key="test-provider-key",
        ),
        transport=httpx.MockTransport(handler),
    )

    sources = adapter.search("research agents", limit=2)

    assert [(source.id, source.title, source.url) for source in sources] == [
        ("S1", "Valid sibling", "https://example.test/valid-report")
    ]
