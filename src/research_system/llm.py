"""Provider-neutral structured model adapters used by the worker agents."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol, TypeVar, cast

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from research_system.config import Settings
from research_system.exceptions import CitationIntegrityError, ConfigurationError
from research_system.models import (
    REPORT_TITLE_MAX_LENGTH,
    ConversationTurn,
    Critique,
    CritiqueDisposition,
    Finding,
    QueryPlan,
    ResearchMode,
    ResearchRequest,
    Source,
    Synthesis,
)
from research_system.prompts import (
    CRITIC_SYSTEM_PROMPT,
    QUERY_PLANNER_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
)

_CITATION_PATTERN = re.compile(r"\[(S[1-9][0-9]*)\]")
_CONTEXT_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9-]{3,}")
_CONTEXT_STOPWORDS = {
    "about",
    "against",
    "available",
    "brief",
    "citation",
    "conclusion",
    "evidence",
    "findings",
    "limitations",
    "quality",
    "report",
    "research",
    "source",
    "sources",
    "summary",
    "supports",
    "using",
}
MODEL_SOURCE_EVIDENCE_CHAR_BUDGET = 6_000
MODEL_TOTAL_EVIDENCE_CHAR_BUDGET = 60_000
StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class ReportDraft(BaseModel):
    """Structured report content before run metadata and provenance are attached."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    title: str = Field(min_length=5, max_length=REPORT_TITLE_MAX_LENGTH)
    executive_summary: str = Field(min_length=20, max_length=10_000)
    markdown: str = Field(min_length=100, max_length=500_000)
    source_ids: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = ()


class ResearchModel(Protocol):
    """Small structured interface shared by deterministic and hosted models."""

    def plan_queries(
        self, request: ResearchRequest, critique: Critique | None = None
    ) -> QueryPlan: ...

    def synthesize(
        self,
        request: ResearchRequest,
        sources: Sequence[Source],
        *,
        critic_feedback: Critique | None = None,
    ) -> Synthesis: ...

    def critique(
        self,
        request: ResearchRequest,
        sources: Sequence[Source],
        synthesis: Synthesis,
    ) -> Critique: ...

    def write_report(
        self,
        request: ResearchRequest,
        sources: Sequence[Source],
        synthesis: Synthesis,
        critique: Critique,
    ) -> ReportDraft: ...


def extract_citation_ids(markdown: str) -> set[str]:
    """Return normalized inline citation IDs found in Markdown."""

    return set(_CITATION_PATTERN.findall(markdown))


def validate_citations(markdown: str, known_ids: set[str]) -> set[str]:
    """Reject missing or unknown evidence references before a report can persist."""

    cited = extract_citation_ids(markdown)
    unknown = cited - known_ids
    if unknown:
        labels = ", ".join(sorted(unknown))
        raise CitationIntegrityError(f"Report contains unknown source IDs: {labels}")
    if known_ids and not cited:
        raise CitationIntegrityError("Report must contain at least one known source citation")
    return cited


def _compact_evidence(source: Source, limit: int = 360) -> str:
    evidence = " ".join(source.evidence_text.split())
    if len(evidence) <= limit:
        return evidence
    return f"{evidence[: limit - 1].rstrip()}…"


def _bounded_report_title(topic: str) -> str:
    """Build a readable report title that always satisfies the shared schema boundary."""

    title = f"Research brief: {topic}"
    if len(title) <= REPORT_TITLE_MAX_LENGTH:
        return title
    return f"{title[: REPORT_TITLE_MAX_LENGTH - 1].rstrip()}…"


def _bounded_prompt_text(value: str, limit: int) -> str:
    """Collapse and truncate evidence before it crosses a hosted-model boundary."""

    if limit <= 0:
        return ""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return f"{text[: limit - 1].rstrip()}…"


def _source_prompt_payloads(sources: Sequence[Source]) -> list[dict[str, object]]:
    """Return metadata-complete source records with fair per-source/aggregate evidence caps."""

    if not sources:
        return []
    source_budget = min(
        MODEL_SOURCE_EVIDENCE_CHAR_BUDGET,
        MODEL_TOTAL_EVIDENCE_CHAR_BUDGET // len(sources),
    )
    records: list[dict[str, object]] = []
    for source in sources:
        snippet_budget = source_budget
        if source.content and source.snippet:
            snippet_budget = min(1_000, max(1, source_budget // 4))
        snippet_excerpt = _bounded_prompt_text(source.snippet, snippet_budget)
        content_excerpt = _bounded_prompt_text(
            source.content,
            max(0, source_budget - len(snippet_excerpt)),
        )
        records.append(
            {
                "id": source.id,
                "title": source.title,
                "url": source.url,
                "provider": source.provider,
                "integrity": source.integrity.value,
                "locator": source.locator,
                "snippet_excerpt": snippet_excerpt,
                "content_excerpt": content_excerpt,
            }
        )
    return records


def _follow_up_answer(
    request: ResearchRequest,
    synthesis: Synthesis,
    known_source_ids: set[str],
) -> str:
    """Answer a deterministic follow-up from current cited findings."""

    if not request.follow_up:
        return ""
    question = " ".join(request.follow_up.split())
    findings: list[tuple[str, tuple[str, ...]]] = []
    for finding in synthesis.findings[:3]:
        source_ids = tuple(
            source_id for source_id in finding.source_ids if source_id in known_source_ids
        )
        if source_ids:
            findings.append((" ".join(finding.claim.split()), source_ids))
    if not findings:
        return ""

    intent_words = set(re.findall(r"[a-z]+", question.casefold()))
    compare_requested = bool(intent_words & {"compare", "comparison", "versus", "vs", "table"})
    if compare_requested:
        rows = [
            "| Evidence perspective | Finding | Sources |",
            "|---|---|---|",
        ]
        for index, (claim, source_ids) in enumerate(findings, start=1):
            safe_claim = claim.replace("|", "\\|")
            citations = " ".join(f"[{source_id}]" for source_id in source_ids)
            rows.append(f"| Perspective {index} | {safe_claim} | {citations} |")
        answer = "\n".join(rows)
    else:
        answer = " ".join(
            f"{claim} {' '.join(f'[{source_id}]' for source_id in source_ids)}"
            for claim, source_ids in findings[:2]
        )
    return f"## Follow-up answer\n\n**Question:** {question}\n\n{answer}\n\n"


def _context_keywords(
    conversation_context: Sequence[ConversationTurn], limit: int = 8
) -> tuple[str, ...]:
    """Extract a small, deterministic set of search anchors from prior assistant output."""

    assistant_text = "\n".join(
        turn.content for turn in conversation_context if turn.role == "assistant"
    )
    text = assistant_text or "\n".join(turn.content for turn in conversation_context)
    candidates: dict[str, tuple[str, int]] = {}
    for index, token in enumerate(_CONTEXT_WORD_PATTERN.findall(text)):
        normalized = token.casefold()
        if normalized in _CONTEXT_STOPWORDS or normalized.startswith("http"):
            continue
        candidates.setdefault(normalized, (token, index))
    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -int(any(character.isupper() for character in item[0][1:])),
            -len(item[0]),
            item[1],
        ),
    )
    return tuple(token.casefold() for token, _ in ranked[:limit])


class DeterministicResearchModel:
    """Truthful, repeatable model for demos, tests, and offline evaluation."""

    def plan_queries(
        self,
        request: ResearchRequest,
        critique: Critique | None = None,
        *,
        conversation_context: Sequence[ConversationTurn] = (),
    ) -> QueryPlan:
        if critique and critique.requested_queries:
            queries = critique.requested_queries
            rationale = "The critic identified specific evidence gaps for bounded revision."
        elif request.follow_up and conversation_context:
            context_keywords = _context_keywords(conversation_context)
            context_anchor = " ".join(context_keywords) or "prior report findings"
            follow_up = " ".join(request.follow_up.split())[:240]
            topic = request.topic[:240]
            queries = (
                f"{topic} {follow_up} prior report context {context_anchor}",
                f"{topic} {follow_up} counterevidence limitations {context_anchor}",
            )
            rationale = (
                "The follow-up plan is anchored to a bounded set of concepts from the prior "
                "thread report, then checks both support and counterevidence."
            )
        else:
            queries = (
                f"{request.topic} evidence and evaluation",
                f"{request.topic} limitations and competing findings",
            )
            rationale = (
                "The plan covers direct evidence plus limitations so the final report can be "
                "balanced and decision-ready."
            )
        return QueryPlan(queries=tuple(queries[:8]), rationale=rationale)

    def synthesize(
        self,
        request: ResearchRequest,
        sources: Sequence[Source],
        *,
        critic_feedback: Critique | None = None,
    ) -> Synthesis:
        if not sources:
            raise ValueError("Synthesis requires at least one source")
        topic = request.topic.rstrip().rstrip(".!?")
        findings = tuple(
            Finding(
                claim=(
                    f"{source.title}: {_compact_evidence(source)}"
                    if source.evidence_text
                    else f"{source.title} is relevant to {request.topic}."
                ),
                source_ids=(source.id,),
                confidence=0.78 if source.score is None else max(0.5, source.score),
            )
            for source in sources[:12]
        )
        executive_summary = (
            f"The available evidence provides {len(findings)} grounded finding"
            f"{'s' if len(findings) != 1 else ''} about {topic}. Together, the sources "
            "support a structured assessment while preserving provider and evidence limits."
        )
        if critic_feedback and critic_feedback.gaps:
            feedback = _bounded_prompt_text("; ".join(critic_feedback.gaps), 600)
            executive_summary += (
                " This revision explicitly responds to the critic's requested clarification: "
                f"{feedback}"
            )
        return Synthesis(
            executive_summary=executive_summary,
            themes=("Evidence", "Quality controls", "Limitations"),
            findings=findings,
            evidence_gaps=(
                "The deterministic demo corpus is illustrative rather than a live "
                "literature review.",
            ),
        )

    def critique(
        self,
        request: ResearchRequest,
        sources: Sequence[Source],
        synthesis: Synthesis,
    ) -> Critique:
        del request
        known = {source.id for source in sources}
        finding_ids = {
            source_id for finding in synthesis.findings for source_id in finding.source_ids
        }
        coverage = len(finding_ids & known) / max(1, len(finding_ids))
        quality = sum(0.9 if source.score is None else source.score for source in sources) / len(
            sources
        )
        return Critique(
            disposition=CritiqueDisposition("approved"),
            overall_score=round((coverage + quality) / 2, 3),
            citation_coverage=round(coverage, 3),
            source_quality=round(quality, 3),
            strengths=(
                "Every synthesized finding maps to an acquired source identifier.",
                "The report can distinguish evidence from stated limitations.",
            ),
            gaps=synthesis.evidence_gaps,
        )

    def write_report(
        self,
        request: ResearchRequest,
        sources: Sequence[Source],
        synthesis: Synthesis,
        critique: Critique,
        *,
        conversation_context: Sequence[ConversationTurn] = (),
    ) -> ReportDraft:
        known = {source.id for source in sources}
        finding_sections: list[str] = []
        used_ids: list[str] = []
        for index, finding in enumerate(synthesis.findings, start=1):
            valid_ids = [source_id for source_id in finding.source_ids if source_id in known]
            if not valid_ids:
                continue
            citations = " ".join(f"[{source_id}]" for source_id in valid_ids)
            used_ids.extend(valid_ids)
            finding_sections.append(f"### {index}. Finding\n\n{finding.claim} {citations}")

        limitations = synthesis.evidence_gaps or (
            "The evidence set is bounded by the configured providers and source budget.",
        )
        limitation_lines = "\n".join(f"- {item}" for item in limitations)
        body = "\n\n".join(finding_sections)
        context_keywords = (
            _context_keywords(conversation_context)
            if request.follow_up and conversation_context
            else ()
        )
        continuity_section = ""
        if context_keywords:
            anchors = ", ".join(f"`{keyword}`" for keyword in context_keywords)
            continuity_section = (
                "## Follow-up continuity\n\n"
                "This follow-up was interpreted against bounded prior-thread context, "
                f"including these search anchors: {anchors}.\n\n"
            )
        follow_up_section = _follow_up_answer(request, synthesis, known)
        markdown = (
            f"# Research brief: {request.topic}\n\n"
            "## Executive summary\n\n"
            f"{synthesis.executive_summary}\n\n"
            "## Evidence-backed findings\n\n"
            f"{body}\n\n"
            "## Quality review\n\n"
            f"The critic scored this synthesis {critique.overall_score:.0%} overall and "
            f"{critique.citation_coverage:.0%} for citation coverage.\n\n"
            f"{follow_up_section}"
            f"{continuity_section}"
            "## Limitations\n\n"
            f"{limitation_lines}\n\n"
            "## Conclusion\n\n"
            "The evidence supports a supervised, citation-checked approach while the stated "
            "limitations define where additional research would be most valuable."
        )
        return ReportDraft(
            title=_bounded_report_title(request.topic),
            executive_summary=synthesis.executive_summary,
            markdown=markdown,
            source_ids=tuple(dict.fromkeys(used_ids)),
            limitations=tuple(limitations),
        )


class OpenAIResearchModel:
    """ChatOpenAI adapter that validates every model response against a schema."""

    def __init__(self, settings: Settings, client: ChatOpenAI | None = None) -> None:
        if settings.openai_api_key is None and client is None:
            raise ConfigurationError("Live mode requires OPENAI_API_KEY")
        self._client = client or ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout_seconds,
            max_retries=2,
        )

    def _invoke(
        self,
        schema: type[StructuredModel],
        system_prompt: str,
        payload: dict[str, object],
    ) -> StructuredModel:
        runnable = self._client.with_structured_output(schema)
        response = runnable.invoke(
            [
                ("system", system_prompt),
                ("human", json.dumps(payload, ensure_ascii=False, default=str)),
            ]
        )
        return cast(StructuredModel, response)

    def plan_queries(
        self,
        request: ResearchRequest,
        critique: Critique | None = None,
        *,
        conversation_context: Sequence[ConversationTurn] = (),
    ) -> QueryPlan:
        return self._invoke(
            QueryPlan,
            QUERY_PLANNER_SYSTEM_PROMPT,
            {
                "request": request.model_dump(mode="json"),
                "critic_feedback": critique.model_dump(mode="json") if critique else None,
                "conversation_context": [
                    {"role": turn.role, "content": turn.content} for turn in conversation_context
                ],
            },
        )

    def synthesize(
        self,
        request: ResearchRequest,
        sources: Sequence[Source],
        *,
        critic_feedback: Critique | None = None,
    ) -> Synthesis:
        return self._invoke(
            Synthesis,
            SUMMARIZER_SYSTEM_PROMPT,
            {
                "request": request.model_dump(mode="json"),
                "sources": _source_prompt_payloads(sources),
                "critic_feedback": (
                    critic_feedback.model_dump(mode="json") if critic_feedback else None
                ),
            },
        )

    def critique(
        self,
        request: ResearchRequest,
        sources: Sequence[Source],
        synthesis: Synthesis,
    ) -> Critique:
        return self._invoke(
            Critique,
            CRITIC_SYSTEM_PROMPT,
            {
                "request": request.model_dump(mode="json"),
                "source_inventory": [
                    {
                        "id": source.id,
                        "title": source.title,
                        "kind": source.kind.value,
                        "integrity": source.integrity.value,
                    }
                    for source in sources
                ],
                "synthesis": synthesis.model_dump(mode="json"),
            },
        )

    def write_report(
        self,
        request: ResearchRequest,
        sources: Sequence[Source],
        synthesis: Synthesis,
        critique: Critique,
        *,
        conversation_context: Sequence[ConversationTurn] = (),
    ) -> ReportDraft:
        return self._invoke(
            ReportDraft,
            WRITER_SYSTEM_PROMPT,
            {
                "request": request.model_dump(mode="json"),
                "sources": [
                    {
                        "id": source.id,
                        "title": source.title,
                        "url": source.url,
                        "locator": source.locator,
                    }
                    for source in sources
                ],
                "synthesis": synthesis.model_dump(mode="json"),
                "critique": critique.model_dump(mode="json"),
                "conversation_context": [
                    {"role": turn.role, "content": turn.content} for turn in conversation_context
                ],
            },
        )


def build_research_model(settings: Settings) -> ResearchModel:
    """Select the deterministic or hosted implementation from validated settings."""

    if settings.research_mode == ResearchMode.DEMO:
        return DeterministicResearchModel()
    return OpenAIResearchModel(settings)
