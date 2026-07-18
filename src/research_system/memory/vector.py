"""Persistent SQLite-backed vector memory for accepted research reports."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import quote

from research_system.memory.embeddings import DeterministicEmbeddingFunction
from research_system.models import IntegrityLabel, ResearchReport, Source, SourceKind

_DATABASE_NAME = "vector-memory.sqlite3"
_SCHEMA_VERSION = 1
_CITATION_TOKEN = re.compile(r"\[S([1-9][0-9]*)\]")


class VectorMemory:
    """Store completed reports locally and recall them as provenance-labeled sources.

    Vectors and their report metadata live in a private SQLite database. Similarity
    is calculated in-process over deterministic normalized embeddings, which keeps
    the keyless path dependency-light and avoids an exposed vector-server surface.
    """

    def __init__(self, path: Path | str) -> None:
        requested_path = Path(path)
        if requested_path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            self.path = requested_path
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        else:
            requested_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            requested_path.chmod(0o700)
            self.path = requested_path / _DATABASE_NAME
        self._embeddings = DeterministicEmbeddingFunction()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 10000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS report_memories (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    provenance_mode TEXT NOT NULL,
                    source_count INTEGER NOT NULL CHECK (source_count >= 0),
                    document TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_report_memories_created_at "
                "ON report_memories(created_at DESC)"
            )
            expected_metadata = {
                "schema_version": str(_SCHEMA_VERSION),
                "embedding_name": self._embeddings.name(),
                "embedding_dimensions": str(self._embeddings.dimensions),
            }
            for key, expected_value in expected_metadata.items():
                row = connection.execute(
                    "SELECT value FROM memory_metadata WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO memory_metadata (key, value) VALUES (?, ?)",
                        (key, expected_value),
                    )
                elif str(row["value"]) != expected_value:
                    raise RuntimeError(
                        f"vector memory {key} is incompatible; create a new index and re-embed"
                    )
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        os.chmod(self.path, 0o600)

    def remember(self, report: ResearchReport, sources: Sequence[Source] = ()) -> None:
        """Upsert one report-level memory record with a compact evidence inventory."""

        inventory = "\n".join(
            f"- {source.id}: {source.title} ({source.integrity.value})" for source in sources[:20]
        )
        document = (
            f"{report.title}\n\n"
            f"Topic: {report.topic}\n\n"
            f"{report.executive_summary}\n\n"
            f"{report.markdown}\n\n"
            f"Evidence inventory:\n{inventory}"
        )[:499_000]
        embedding_json = json.dumps(
            self._embeddings([document])[0], separators=(",", ":"), allow_nan=False
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO report_memories (
                    run_id, thread_id, topic, title, created_at, provenance_mode,
                    source_count, document, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    topic = excluded.topic,
                    title = excluded.title,
                    created_at = excluded.created_at,
                    provenance_mode = excluded.provenance_mode,
                    source_count = excluded.source_count,
                    document = excluded.document,
                    embedding_json = excluded.embedding_json
                """,
                (
                    report.run_id,
                    report.thread_id,
                    report.topic,
                    report.title,
                    report.created_at.isoformat(),
                    report.provenance_mode.value,
                    len(sources),
                    document,
                    embedding_json,
                ),
            )

    def recall(self, query: str, limit: int = 3) -> list[Source]:
        """Return the most relevant prior reports as accepted-memory evidence."""

        if limit <= 0 or not query.strip():
            return []
        query_vector = self._embeddings([query])[0]
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, title, document, embedding_json FROM report_memories"
            ).fetchall()

        ranked: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            vector = self._decode_vector(row["embedding_json"])
            if vector is None:
                continue
            similarity = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            ranked.append((min(max(similarity, 0.0), 1.0), row))
        ranked.sort(key=lambda item: (-item[0], str(item[1]["run_id"])))

        recalled: list[Source] = []
        for index, (score, row) in enumerate(ranked[: min(limit, 50)], 1):
            run_id = str(row["run_id"])
            document = _CITATION_TOKEN.sub(r"[prior-S\1]", str(row["document"]))
            recalled.append(
                Source(
                    id=f"S{index}",
                    kind=SourceKind.MEMORY,
                    title=str(row["title"] or "Prior research report")[:500],
                    url=f"memory://{quote(run_id, safe='')}",
                    snippet=document[:20_000],
                    content=document[:500_000],
                    provider="local SQLite vector memory",
                    integrity=IntegrityLabel.ACCEPTED_MEMORY,
                    score=score,
                    locator=run_id[:500],
                    checksum=sha256(document.encode("utf-8")).hexdigest(),
                )
            )
        return recalled

    def _decode_vector(self, raw_vector: Any) -> list[float] | None:
        try:
            decoded = json.loads(str(raw_vector))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(decoded, list) or len(decoded) != self._embeddings.dimensions:
            return None
        if not all(isinstance(value, int | float) for value in decoded):
            return None
        return [float(value) for value in decoded]
