"""Typed health, research, and durable run-readback routes."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query, Request, status

from research_system.engine import ResearchEngine
from research_system.models import ResearchRequest, ResearchResult, RunStatus

router = APIRouter()


def _engine(request: Request) -> ResearchEngine:
    return cast(ResearchEngine, request.app.state.engine)


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    engine = _engine(request)
    return {"status": "ok", "readiness": engine.settings.readiness()}


@router.post("/research", response_model=ResearchResult)
def research(
    payload: ResearchRequest,
    request: Request,
    thread_id: str | None = Query(default=None, min_length=1, max_length=200),
) -> ResearchResult:
    result = _engine(request).run(payload, thread_id=thread_id)
    if result.status == RunStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"message": "Research workflow failed.", "run_id": result.run_id},
        )
    return result


@router.get("/runs", response_model=list[ResearchResult])
def list_runs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[ResearchResult]:
    return _engine(request).list_runs(limit=limit)


@router.get("/runs/{run_id}", response_model=ResearchResult)
def get_run(run_id: str, request: Request) -> ResearchResult:
    result = _engine(request).get_run(run_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research run not found.",
        )
    return result
