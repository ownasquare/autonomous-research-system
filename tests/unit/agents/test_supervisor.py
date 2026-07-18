from __future__ import annotations

from research_system.agents.supervisor import choose_route, supervisor_node


def _request(max_revisions: int = 1) -> dict[str, object]:
    return {
        "topic": "How should multi-agent research systems be evaluated?",
        "max_revisions": max_revisions,
    }


def test_supervisor_routes_workers_in_required_order() -> None:
    assert choose_route({"request": _request()}) == "researcher"
    assert choose_route({"request": _request(), "sources": [{"id": "S1"}]}) == "summarizer"
    assert (
        choose_route(
            {
                "request": _request(),
                "sources": [{"id": "S1"}],
                "synthesis": {"executive_summary": "Evidence is organized."},
            }
        )
        == "critic"
    )
    assert (
        choose_route(
            {
                "request": _request(),
                "sources": [{"id": "S1"}],
                "synthesis": {"executive_summary": "Evidence is organized."},
                "critique": {"disposition": "approved"},
            }
        )
        == "writer"
    )


def test_supervisor_honors_revision_budget_then_writes() -> None:
    state = {
        "request": _request(max_revisions=1),
        "sources": [{"id": "S1"}],
        "synthesis": {"executive_summary": "Evidence is organized."},
        "critique": {"disposition": "revise_research"},
        "revision_count": 0,
    }
    assert choose_route(state) == "researcher"
    state["revision_count"] = 1
    assert choose_route(state) == "writer"


def test_supervisor_warns_when_revision_budget_is_exhausted() -> None:
    state = {
        "request": _request(max_revisions=1),
        "sources": [{"id": "S1"}],
        "synthesis": {"executive_summary": "Evidence is organized."},
        "critique": {"disposition": "revise_summary"},
        "revision_count": 1,
    }

    update = supervisor_node(state)  # type: ignore[arg-type]

    assert update["route"] == "writer"
    assert update["warnings"]
    assert "revision budget (1) was exhausted" in update["warnings"][0]


def test_supervisor_ends_after_report_or_error() -> None:
    assert choose_route({"request": _request(), "report": {"markdown": "done"}}) == "end"
    assert choose_route({"request": _request(), "error": "failed"}) == "end"
