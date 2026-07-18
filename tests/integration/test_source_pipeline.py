from __future__ import annotations

from hashlib import sha256

import pymupdf
import pytest

from research_system.config import Settings
from research_system.exceptions import ProviderError
from research_system.models import (
    IntegrityLabel,
    ResearchMode,
    ResearchRequest,
    Source,
    SourceKind,
)
from research_system.tools.sources import SourcePipeline


def _source(source_id: str, url: str, content: str = "Uploaded evidence") -> Source:
    return Source(
        id=source_id,
        kind=SourceKind.PDF,
        title="Uploaded source",
        url=url,
        content=content,
        provider="test upload",
        integrity=IntegrityLabel.USER_UPLOAD,
        checksum=sha256(content.encode()).hexdigest(),
    )


def _provider_source(kind: SourceKind, label: str) -> Source:
    content = f"Evidence from {label}"
    integrity = {
        SourceKind.WEB: IntegrityLabel.LIVE_WEB,
        SourceKind.ARXIV: IntegrityLabel.LIVE_ARXIV,
        SourceKind.MEMORY: IntegrityLabel.ACCEPTED_MEMORY,
    }[kind]
    scheme = "memory" if kind is SourceKind.MEMORY else "https"
    return Source(
        id="S1",
        kind=kind,
        title=f"{label} source",
        url=f"{scheme}://evidence.test/{label}",
        content=content,
        provider=f"test {kind.value}",
        integrity=integrity,
        checksum=sha256(content.encode()).hexdigest(),
    )


class StubMemory:
    def __init__(self, sources: list[Source]) -> None:
        self.sources = sources
        self.calls: list[tuple[str, int]] = []

    def recall(self, query: str, limit: int = 3) -> list[Source]:
        self.calls.append((query, limit))
        return self.sources[:limit]


def test_demo_mode_uses_bundled_fixture_and_never_live_adapters(tmp_path) -> None:
    settings = Settings(research_mode=ResearchMode.DEMO, research_data_dir=tmp_path)
    pipeline = SourcePipeline(settings)
    pipeline._tavily.search = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("live web adapter was called")
    )
    pipeline._arxiv.search = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("live arXiv adapter was called")
    )

    sources, warnings = pipeline.gather(ResearchRequest(topic="agent orchestration"), ["agents"])

    assert sources
    assert not warnings
    assert {source.integrity for source in sources} == {IntegrityLabel.DEMO_FIXTURE}


def test_demo_mode_memory_off_is_fixture_only_and_does_not_read_state(tmp_path) -> None:
    memory = StubMemory([_provider_source(SourceKind.MEMORY, "prior-report")])
    pipeline = SourcePipeline(
        Settings(research_mode=ResearchMode.DEMO, research_data_dir=tmp_path),
        memory=memory,
    )

    sources, warnings = pipeline.gather(
        ResearchRequest(topic="agent orchestration", use_memory=False),
        ["agents"],
    )

    assert not warnings
    assert not memory.calls
    assert {source.integrity for source in sources} == {IntegrityLabel.DEMO_FIXTURE}


def test_demo_mode_memory_on_fairly_includes_accepted_prior_reports(tmp_path) -> None:
    memory = StubMemory([_provider_source(SourceKind.MEMORY, "prior-report")])
    pipeline = SourcePipeline(
        Settings(
            research_mode=ResearchMode.DEMO,
            research_data_dir=tmp_path,
            max_search_results=2,
        ),
        memory=memory,
    )

    sources, warnings = pipeline.gather(
        ResearchRequest(topic="agent orchestration", max_sources=4, use_memory=True),
        ["agents"],
    )

    assert not warnings
    assert memory.calls == [("agent orchestration", 2)]
    assert [source.kind for source in sources[:2]] == [SourceKind.DEMO, SourceKind.MEMORY]
    assert [source.id for source in sources] == ["S1", "S2", "S3", "S4"]


def test_demo_mode_memory_only_returns_exactly_selected_memory(tmp_path) -> None:
    memory_source = _provider_source(SourceKind.MEMORY, "prior-report")
    memory = StubMemory([memory_source])
    pipeline = SourcePipeline(
        Settings(research_mode=ResearchMode.DEMO, research_data_dir=tmp_path),
        memory=memory,
    )

    sources, warnings = pipeline.gather(
        ResearchRequest(
            topic="agent orchestration",
            use_demo=False,
            use_memory=True,
        ),
        ["agents"],
    )

    assert not warnings
    assert [source.kind for source in sources] == [SourceKind.MEMORY]
    assert [source.title for source in sources] == [memory_source.title]
    assert [source.id for source in sources] == ["S1"]


def test_demo_mode_pdf_only_returns_exactly_selected_upload(tmp_path) -> None:
    pipeline = SourcePipeline(Settings(research_mode=ResearchMode.DEMO, research_data_dir=tmp_path))
    uploaded = _source("S9", "upload://notes.pdf")

    sources, warnings = pipeline.gather(
        ResearchRequest(
            topic="agent orchestration",
            use_demo=False,
            use_memory=False,
        ),
        ["agents"],
        uploaded_sources=[uploaded],
    )

    assert not warnings
    assert [source.kind for source in sources] == [SourceKind.PDF]
    assert [source.title for source in sources] == [uploaded.title]
    assert [source.id for source in sources] == ["S1"]


@pytest.mark.parametrize(
    "topic",
    [
        "Medieval Icelandic sheep genetics",
        "Which energy source should California prioritize?",
        "How should a payroll workflow be redesigned?",
        "How does computer memory affect gaming performance?",
        "Which talent agent should I hire?",
    ],
)
def test_demo_mode_rejects_unrelated_fixture_substitution(tmp_path, topic: str) -> None:
    pipeline = SourcePipeline(Settings(research_mode=ResearchMode.DEMO, research_data_dir=tmp_path))
    request = ResearchRequest(
        topic=topic,
        use_memory=False,
    )

    with pytest.raises(ProviderError, match="bundled demo covers"):
        pipeline.gather(request, [topic])


def test_demo_mode_allows_unrelated_topic_when_pdf_evidence_is_attached(tmp_path) -> None:
    pipeline = SourcePipeline(Settings(research_mode=ResearchMode.DEMO, research_data_dir=tmp_path))
    request = ResearchRequest(
        topic="Medieval Icelandic sheep genetics",
        use_memory=False,
    )
    uploaded = _source("S9", "upload://sheep-notes.pdf", "Genetic evidence from the PDF")

    sources, warnings = pipeline.gather(request, [], uploaded_sources=[uploaded])

    assert [source.kind for source in sources] == [SourceKind.PDF]
    assert warnings == ["Bundled fixtures were skipped because they do not cover this topic."]


def test_live_mode_never_silently_injects_demo_evidence(tmp_path) -> None:
    settings = Settings(research_mode=ResearchMode.LIVE, research_data_dir=tmp_path)
    request = ResearchRequest(
        topic="agent orchestration",
        use_web=False,
        use_arxiv=False,
        use_memory=False,
    )

    sources, warnings = SourcePipeline(settings).gather(request, ["agents"])

    assert sources == []
    assert warnings == []


def test_uploaded_sources_survive_disabled_providers_and_ids_are_normalized(tmp_path) -> None:
    settings = Settings(research_mode=ResearchMode.LIVE, research_data_dir=tmp_path)
    request = ResearchRequest(
        topic="agent orchestration",
        use_web=False,
        use_arxiv=False,
        use_memory=False,
    )
    uploaded = _source("S9", "upload://notes.pdf")

    sources, _ = SourcePipeline(settings).gather(request, [], uploaded_sources=[uploaded])

    assert [source.id for source in sources] == ["S1"]
    assert sources[0].integrity is IntegrityLabel.USER_UPLOAD


def test_pipeline_deduplicates_canonical_urls(tmp_path) -> None:
    pipeline = SourcePipeline(Settings(research_data_dir=tmp_path))
    first = _source("S1", "https://EXAMPLE.com/paper?utm_source=test&b=2&a=1")
    duplicate = _source("S9", "https://example.com/paper?a=1&b=2#abstract")

    sources = pipeline.deduplicate([first, duplicate])

    assert [source.id for source in sources] == ["S1"]


def test_live_pipeline_fairly_interleaves_providers_with_uploaded_sources_first(
    tmp_path,
) -> None:
    memory = StubMemory(
        [
            _provider_source(SourceKind.MEMORY, "memory-1"),
            _provider_source(SourceKind.MEMORY, "memory-2"),
        ]
    )
    pipeline = SourcePipeline(
        Settings(
            research_mode=ResearchMode.LIVE,
            research_data_dir=tmp_path,
            tavily_api_key="test-provider-key",
            max_search_results=2,
        ),
        memory=memory,
    )
    calls: list[tuple[str, str, int]] = []

    def web_search(query: str, limit: int) -> list[Source]:
        calls.append(("web", query, limit))
        return [_provider_source(SourceKind.WEB, f"{query}-web-{index}") for index in range(1, 3)]

    def arxiv_search(query: str, limit: int) -> list[Source]:
        calls.append(("arxiv", query, limit))
        return [
            _provider_source(SourceKind.ARXIV, f"{query}-arxiv-{index}") for index in range(1, 3)
        ]

    pipeline._tavily.search = web_search  # type: ignore[method-assign]
    pipeline._arxiv.search = arxiv_search  # type: ignore[method-assign]
    uploaded = _source("S9", "upload://notes.pdf")

    sources, warnings = pipeline.gather(
        ResearchRequest(topic="agent orchestration", max_sources=6),
        ["query-a", "query-b"],
        uploaded_sources=[uploaded],
    )

    assert not warnings
    assert calls == [
        ("web", "query-a", 2),
        ("web", "query-b", 2),
        ("arxiv", "query-a", 2),
        ("arxiv", "query-b", 2),
    ]
    assert memory.calls == [("agent orchestration", 2)]
    assert [source.kind for source in sources] == [
        SourceKind.PDF,
        SourceKind.WEB,
        SourceKind.ARXIV,
        SourceKind.MEMORY,
        SourceKind.WEB,
        SourceKind.ARXIV,
    ]
    assert [source.id for source in sources] == ["S1", "S2", "S3", "S4", "S5", "S6"]


def test_parse_uploads_preserves_tuple_filename(tmp_path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Tuple upload evidence")
    data = document.tobytes()
    document.close()

    sources, warnings = SourcePipeline(Settings(research_data_dir=tmp_path)).parse_uploads(
        [("named-upload.pdf", data)]
    )

    assert not warnings
    assert sources[0].title == "named-upload.pdf"
    assert sources[0].url == "upload://named-upload.pdf"


def test_parse_uploads_enforces_count_and_aggregate_byte_budgets(tmp_path) -> None:
    class RecordingParser:
        def __init__(self) -> None:
            self.filenames: list[str] = []

        def parse(self, data: bytes, filename: str) -> list[Source]:
            self.filenames.append(filename)
            return [_source("S1", f"upload://{filename}", data[:16].hex())]

    pipeline = SourcePipeline(
        Settings(
            research_data_dir=tmp_path,
            max_pdf_bytes=100_000,
            max_pdf_uploads=2,
            max_upload_bytes_total=100_000,
        )
    )
    parser = RecordingParser()
    pipeline._pdf = parser

    sources, warnings = pipeline.parse_uploads(
        [
            ("first.pdf", b"a" * 60_000),
            ("second.pdf", b"b" * 60_000),
            ("third.pdf", b"c" * 10_000),
        ]
    )

    assert parser.filenames == ["first.pdf"]
    assert len(sources) == 1
    assert warnings == [
        "Only the first 2 PDF uploads were considered.",
        "second.pdf: aggregate PDF upload byte limit exceeded",
    ]
