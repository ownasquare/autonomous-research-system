from __future__ import annotations

import stat

from research_system.models import ResearchRequest, ResearchResult, RunStatus
from research_system.persistence.runs import RunRepository


def test_run_round_trips_and_updates(tmp_path) -> None:
    repository = RunRepository(tmp_path / "runs.sqlite3")
    pending = ResearchResult(
        run_id="run-1",
        thread_id="thread-1",
        request=ResearchRequest(topic="agent orchestration"),
    )
    running = pending.model_copy(update={"status": RunStatus.RUNNING})

    repository.save(pending)
    repository.save(running)

    assert stat.S_IMODE(repository.path.stat().st_mode) == 0o600
    assert repository.get("run-1") == running
    assert repository.list_runs() == [running]


def test_repository_persists_across_reopen_and_limits_results(tmp_path) -> None:
    path = tmp_path / "nested" / "runs.sqlite3"
    repository = RunRepository(path)
    for index in range(3):
        repository.save(
            ResearchResult(
                run_id=f"run-{index}",
                thread_id=f"thread-{index}",
                request=ResearchRequest(topic=f"agent orchestration topic {index}"),
            )
        )

    reopened = RunRepository(path)

    assert len(reopened.list_runs(limit=2)) == 2
    assert reopened.get("missing") is None
