"""Canonical validated records shared by tools, agents, storage, API, and UI."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPORT_TITLE_MAX_LENGTH = 500


def utc_now() -> datetime:
    return datetime.now(UTC)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ResearchMode(StrEnum):
    DEMO = "demo"
    LIVE = "live"


class ResearchDepth(StrEnum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class SourceKind(StrEnum):
    WEB = "web"
    ARXIV = "arxiv"
    PDF = "pdf"
    MEMORY = "memory"
    DEMO = "demo"


class IntegrityLabel(StrEnum):
    LIVE_WEB = "live_web"
    LIVE_ARXIV = "live_arxiv"
    USER_UPLOAD = "user_upload"
    ACCEPTED_MEMORY = "accepted_memory"
    DEMO_FIXTURE = "demo_fixture"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CritiqueDisposition(StrEnum):
    APPROVED = "approved"
    REVISE_RESEARCH = "revise_research"
    REVISE_SUMMARY = "revise_summary"


class ResearchRequest(FrozenModel):
    topic: str = Field(min_length=5, max_length=500)
    objective: str = Field(
        default="Produce a balanced, decision-ready research report.",
        min_length=5,
        max_length=1_000,
    )
    audience: str = Field(default="General professional reader", min_length=3, max_length=200)
    depth: ResearchDepth = ResearchDepth.STANDARD
    use_web: bool = True
    use_arxiv: bool = True
    use_memory: bool = True
    use_demo: bool = True
    max_sources: int = Field(default=12, ge=2, le=20)
    max_revisions: int = Field(default=1, ge=0, le=2)
    follow_up: str | None = Field(default=None, min_length=5, max_length=1_000)

    @field_validator("topic", "objective", "audience")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value.strip()

    @model_validator(mode="before")
    @classmethod
    def apply_depth_defaults(cls, data: Any) -> Any:
        """Choose sensible budgets when callers select a depth without overriding them."""

        if not isinstance(data, dict):
            return data
        values = dict(data)
        depth = ResearchDepth(values.get("depth", ResearchDepth.STANDARD))
        defaults = {
            ResearchDepth.QUICK: (6, 0),
            ResearchDepth.STANDARD: (12, 1),
            ResearchDepth.DEEP: (20, 2),
        }
        max_sources, max_revisions = defaults[depth]
        values.setdefault("max_sources", max_sources)
        values.setdefault("max_revisions", max_revisions)
        return values

    @model_validator(mode="after")
    def normalize_depth_budget(self) -> ResearchRequest:
        limits = {
            ResearchDepth.QUICK: (6, 0),
            ResearchDepth.STANDARD: (12, 1),
            ResearchDepth.DEEP: (20, 2),
        }
        max_sources, max_revisions = limits[self.depth]
        if self.max_sources > max_sources:
            raise ValueError(f"{self.depth.value} research supports at most {max_sources} sources")
        if self.max_revisions > max_revisions:
            raise ValueError(
                f"{self.depth.value} research supports at most {max_revisions} revisions"
            )
        return self


class Source(FrozenModel):
    id: str = Field(pattern=r"^S[1-9][0-9]*$")
    kind: SourceKind
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=5, max_length=2_000)
    snippet: str = Field(default="", max_length=20_000)
    content: str = Field(default="", max_length=500_000)
    authors: tuple[str, ...] = ()
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    provider: str = Field(min_length=2, max_length=100)
    integrity: IntegrityLabel
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    locator: str | None = Field(default=None, max_length=500)
    checksum: str = Field(min_length=16, max_length=128)

    @field_validator("url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        if any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        ):
            raise ValueError("source URL cannot contain whitespace or control characters")
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("source URL is malformed") from exc
        allowed_schemes = {"https", "http", "upload", "memory", "demo"}
        scheme = parsed.scheme.casefold()
        if scheme not in allowed_schemes:
            raise ValueError("source URL uses an unsupported scheme")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("source URL cannot include user information")
        if not parsed.hostname:
            raise ValueError("source URL must include a host or workspace identifier")
        try:
            host = parsed.hostname.encode("idna").decode("ascii").casefold()
        except UnicodeError as exc:
            raise ValueError("source URL host is invalid") from exc
        if ":" in host:
            host = f"[{host}]"
        netloc = f"{host}:{port}" if port is not None else host
        path = quote(parsed.path, safe="/%:@-._~")
        query = quote(parsed.query, safe="=&;%:@,+-._~/?")
        fragment = quote(parsed.fragment, safe="=&;%:@,+-._~/?")
        return urlunsplit((scheme, netloc, path, query, fragment))

    @property
    def evidence_text(self) -> str:
        return self.content or self.snippet


class QueryPlan(FrozenModel):
    queries: tuple[str, ...] = Field(min_length=1, max_length=8)
    rationale: str = Field(min_length=5, max_length=2_000)


class Finding(FrozenModel):
    claim: str = Field(min_length=10, max_length=5_000)
    source_ids: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    contradiction: str | None = Field(default=None, max_length=2_000)


class Synthesis(FrozenModel):
    executive_summary: str = Field(min_length=20, max_length=10_000)
    themes: tuple[str, ...] = Field(min_length=1, max_length=12)
    findings: tuple[Finding, ...] = Field(min_length=1, max_length=30)
    evidence_gaps: tuple[str, ...] = ()


class Critique(FrozenModel):
    disposition: CritiqueDisposition
    overall_score: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    source_quality: float = Field(ge=0.0, le=1.0)
    strengths: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    requested_queries: tuple[str, ...] = ()


class AgentTraceEvent(FrozenModel):
    sequence: int = Field(ge=1)
    agent: Literal["supervisor", "researcher", "summarizer", "critic", "writer"]
    status: Literal["started", "completed", "warning", "failed"]
    message: str = Field(min_length=1, max_length=2_000)
    created_at: datetime = Field(default_factory=utc_now)
    details: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ResearchReport(FrozenModel):
    run_id: str
    thread_id: str
    topic: str
    title: str = Field(min_length=5, max_length=REPORT_TITLE_MAX_LENGTH)
    executive_summary: str = Field(min_length=20, max_length=10_000)
    markdown: str = Field(min_length=100, max_length=500_000)
    source_ids: tuple[str, ...] = Field(min_length=1)
    critique: Critique
    limitations: tuple[str, ...] = ()
    provenance_mode: ResearchMode
    created_at: datetime = Field(default_factory=utc_now)


class ConversationTurn(FrozenModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=50_000)
    created_at: datetime = Field(default_factory=utc_now)


class ResearchResult(FrozenModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    thread_id: str = Field(default_factory=lambda: str(uuid4()))
    request: ResearchRequest
    status: RunStatus = RunStatus.PENDING
    sources: tuple[Source, ...] = ()
    report: ResearchReport | None = None
    trace: tuple[AgentTraceEvent, ...] = ()
    conversation: tuple[ConversationTurn, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> ResearchResult:
        completed = {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS}
        if self.status in completed and self.report is None:
            raise ValueError("completed research requires a report")
        if self.status == RunStatus.FAILED and not self.error:
            raise ValueError("failed research requires an error")
        return self


class WorkflowEvent(FrozenModel):
    run_id: str
    stage: Literal["gathering", "organizing", "reviewing", "writing", "complete", "failed"]
    agent: str
    message: str
    progress: float = Field(ge=0.0, le=1.0)
    result: ResearchResult | None = None
    created_at: datetime = Field(default_factory=utc_now)
