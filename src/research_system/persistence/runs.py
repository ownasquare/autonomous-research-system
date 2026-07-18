"""SQLite lifecycle storage for canonical research results."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from threading import RLock
from types import TracebackType

from research_system.models import ResearchResult


class RunRepository:
    """Persist complete validated result snapshots with transactional upserts."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._lock = RLock()
        self._closed = False
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS runs_updated_at_idx ON runs(updated_at DESC)"
            )
        os.chmod(self.path, 0o600)

    def save(self, result: ResearchResult) -> None:
        payload = result.model_dump_json()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runs (
                    run_id, thread_id, status, topic, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    status = excluded.status,
                    topic = excluded.topic,
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    result.run_id,
                    result.thread_id,
                    result.status.value,
                    result.request.topic,
                    payload,
                    result.created_at.isoformat(),
                    result.updated_at.isoformat(),
                ),
            )

    def get(self, run_id: str) -> ResearchResult | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return ResearchResult.model_validate_json(row[0])

    def list_runs(self, limit: int = 50) -> list[ResearchResult]:
        if limit <= 0:
            return []
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM runs ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                (min(limit, 500),),
            ).fetchall()
        return [ResearchResult.model_validate_json(row[0]) for row in rows]

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> RunRepository:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
