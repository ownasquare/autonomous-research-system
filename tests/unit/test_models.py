import pytest
from pydantic import ValidationError

from research_system.models import ResearchDepth, ResearchRequest, Source


def test_research_request_rejects_blank_topic() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(topic="   ")


def test_quick_depth_rejects_excessive_budget() -> None:
    with pytest.raises(ValidationError, match="at most 6 sources"):
        ResearchRequest(topic="A valid research topic", depth=ResearchDepth.QUICK, max_sources=7)


def test_depth_selects_bounded_default_budgets() -> None:
    quick = ResearchRequest(topic="A valid research topic", depth=ResearchDepth.QUICK)
    standard = ResearchRequest(topic="A valid research topic")
    deep = ResearchRequest(topic="A valid research topic", depth=ResearchDepth.DEEP)
    assert (quick.max_sources, quick.max_revisions) == (6, 0)
    assert (standard.max_sources, standard.max_revisions) == (12, 1)
    assert (deep.max_sources, deep.max_revisions) == (20, 2)
    schema = ResearchRequest.model_json_schema()
    assert schema["properties"]["max_sources"]["default"] == 12


def test_follow_up_requires_a_substantive_question_when_present() -> None:
    with pytest.raises(ValidationError, match="at least 5 characters"):
        ResearchRequest(topic="A valid research topic", follow_up="why")


def test_source_rejects_unknown_url_scheme() -> None:
    with pytest.raises(ValidationError, match="unsupported scheme"):
        Source(
            id="S1",
            kind="web",
            title="Unsafe source",
            url="file:///private/document",
            provider="test",
            integrity="live_web",
            checksum="0123456789abcdef",
        )


def test_source_url_rejects_markdown_injection_and_userinfo() -> None:
    base = {
        "id": "S1",
        "kind": "web",
        "title": "A source",
        "provider": "test provider",
        "integrity": "live_web",
        "checksum": "a" * 64,
    }

    with pytest.raises(ValidationError, match="whitespace or control"):
        Source(url="https://good.example/path\n\n![track](https://evil.example/pixel)", **base)
    with pytest.raises(ValidationError, match="user information"):
        Source(url="https://user:password@example.com/report", **base)

    source = Source(url="HTTPS://Example.COM/report!(draft)", **base)
    assert source.url == "https://example.com/report%21%28draft%29"
