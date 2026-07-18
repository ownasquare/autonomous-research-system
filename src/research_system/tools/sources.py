"""Source acquisition facade shared by the workflow and UI."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol

from research_system.config import Settings
from research_system.exceptions import ProviderError, SourceValidationError
from research_system.models import ResearchMode, ResearchRequest, Source
from research_system.tools.arxiv import ArxivSearch
from research_system.tools.base import deduplicate_sources
from research_system.tools.demo import DemoSearch
from research_system.tools.pdf import PdfParser
from research_system.tools.tavily import TavilySearch


class MemoryReader(Protocol):
    def recall(self, query: str, limit: int = 3) -> list[Source]: ...


class SourcePipeline:
    """Combine uploads, configured providers, and accepted long-term memory."""

    def __init__(self, settings: Settings, memory: MemoryReader | None = None) -> None:
        self._settings = settings
        self._memory = memory
        self._pdf = PdfParser(settings)
        self._tavily = TavilySearch(settings)
        self._arxiv = ArxivSearch(settings)
        self._demo = DemoSearch()

    def parse_uploads(self, uploads: Iterable[object]) -> tuple[list[Source], list[str]]:
        sources: list[Source] = []
        warnings: list[str] = []
        upload_items = tuple(uploads)
        if len(upload_items) > self._settings.max_pdf_uploads:
            warnings.append(
                f"Only the first {self._settings.max_pdf_uploads} PDF uploads were considered."
            )
            upload_items = upload_items[: self._settings.max_pdf_uploads]
        aggregate_bytes = 0
        for index, upload in enumerate(upload_items, 1):
            if isinstance(upload, Source):
                sources.append(upload)
                continue
            tuple_name = upload[0] if isinstance(upload, tuple) and len(upload) == 2 else None
            filename = str(tuple_name or getattr(upload, "name", f"upload-{index}.pdf"))
            try:
                data = _upload_bytes(upload)
                if aggregate_bytes + len(data) > self._settings.max_upload_bytes_total:
                    warnings.append(f"{filename}: aggregate PDF upload byte limit exceeded")
                    continue
                aggregate_bytes += len(data)
                sources.extend(self._pdf.parse(data, filename))
            except (SourceValidationError, TypeError, ValueError) as exc:
                warnings.append(f"{filename}: {exc}")
        return self.deduplicate(sources), warnings

    def gather(
        self,
        request: ResearchRequest,
        queries: Sequence[str],
        uploaded_sources: Sequence[Source] = (),
    ) -> tuple[list[Source], list[str]]:
        sources = list(uploaded_sources)
        warnings: list[str] = []
        normalized_queries = [query.strip() for query in queries if query.strip()] or [
            request.topic
        ]

        if self._settings.research_mode is ResearchMode.DEMO:
            demo_pools: list[list[Source]] = []
            if request.use_demo:
                if self._demo.supports(request.topic):
                    demo_pools.append(
                        self._demo.search(" ".join(normalized_queries), request.max_sources)
                    )
                else:
                    warnings.append(
                        "Bundled fixtures were skipped because they do not cover this topic."
                    )
            if request.use_memory and self._memory is not None:
                try:
                    memory_sources = _relevant_memories(
                        self._memory.recall(
                            request.topic,
                            limit=min(3, self._settings.max_search_results, request.max_sources),
                        )
                    )
                    if memory_sources:
                        demo_pools.append(memory_sources)
                except Exception:
                    warnings.append("Long-term memory could not be read for this run")
            sources.extend(_fair_interleave(demo_pools))
            if not sources:
                if request.use_demo:
                    raise ProviderError(
                        "The bundled demo covers multi-agent research workflows and quality. "
                        "Use live mode or attach a relevant PDF for other topics."
                    )
                raise ProviderError("No usable evidence was acquired from the selected sources")
            return self._finalize(sources, request.max_sources), warnings

        provider_limit = min(self._settings.max_search_results, request.max_sources)
        provider_pools: list[list[Source]] = []

        if request.use_web:
            if self._settings.tavily_api_key is None:
                warnings.append("Tavily search was requested but is not configured")
            else:
                web_batches: list[list[Source]] = []
                for query in normalized_queries:
                    try:
                        web_batches.append(self._tavily.search(query, provider_limit))
                    except ProviderError as exc:
                        warnings.append(f"Tavily search warning for '{query}': {exc}")
                web_sources = _fair_interleave(web_batches)
                if web_sources:
                    provider_pools.append(web_sources)

        if request.use_arxiv:
            arxiv_batches: list[list[Source]] = []
            for query in normalized_queries:
                try:
                    arxiv_batches.append(self._arxiv.search(query, provider_limit))
                except ProviderError as exc:
                    warnings.append(f"arXiv search warning for '{query}': {exc}")
            arxiv_sources = _fair_interleave(arxiv_batches)
            if arxiv_sources:
                provider_pools.append(arxiv_sources)

        if request.use_memory and self._memory is not None:
            try:
                memory_sources = _relevant_memories(
                    self._memory.recall(
                        request.topic,
                        limit=min(3, provider_limit),
                    )
                )
                if memory_sources:
                    provider_pools.append(memory_sources)
            except Exception:
                warnings.append("Long-term memory could not be read for this run")

        sources.extend(_fair_interleave(provider_pools))
        return self._finalize(sources, request.max_sources), warnings

    def deduplicate(self, sources: Sequence[Source]) -> list[Source]:
        return deduplicate_sources(list(sources))

    def _finalize(self, sources: list[Source], limit: int) -> list[Source]:
        return self.deduplicate(sources)[:limit]


def _fair_interleave(groups: Sequence[Sequence[Source]]) -> list[Source]:
    """Round-robin source groups so no configured provider consumes the budget first."""

    pending: list[Iterator[Source]] = [iter(group) for group in groups if group]
    interleaved: list[Source] = []
    while pending:
        next_round: list[Iterator[Source]] = []
        for iterator in pending:
            try:
                interleaved.append(next(iterator))
            except StopIteration:
                continue
            next_round.append(iterator)
        pending = next_round
    return interleaved


def _relevant_memories(sources: Sequence[Source], minimum_score: float = 0.1) -> list[Source]:
    """Discard explicitly low-similarity memory leads while retaining provider stubs."""

    return [source for source in sources if source.score is None or source.score >= minimum_score]


def _upload_bytes(upload: object) -> bytes:
    if isinstance(upload, bytes):
        return upload
    if isinstance(upload, tuple) and len(upload) == 2 and isinstance(upload[1], bytes):
        return upload[1]
    getvalue = getattr(upload, "getvalue", None)
    if callable(getvalue):
        value = getvalue()
        if isinstance(value, bytes):
            return value
    read = getattr(upload, "read", None)
    if callable(read):
        value = read()
        if isinstance(value, bytes):
            return value
    raise TypeError("upload does not provide PDF bytes")
