from __future__ import annotations

import sqlite3
import stat
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from research_system.persistence.checkpoints import SqliteCheckpointManager


class _UntrustedCheckpointType(BaseModel):
    value: str = "must not revive as a Python object"


def test_checkpoint_manager_initializes_and_reopens_database(tmp_path) -> None:
    path = tmp_path / "nested" / "checkpoints.sqlite3"

    with SqliteCheckpointManager(path) as manager:
        assert isinstance(manager.saver, SqliteSaver)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    assert {"checkpoints", "writes"} <= tables


def test_checkpoint_close_is_idempotent(tmp_path) -> None:
    manager = SqliteCheckpointManager(tmp_path / "checkpoints.sqlite3")

    manager.close()


def test_checkpoint_serializer_blocks_unlisted_python_types(tmp_path) -> None:
    with SqliteCheckpointManager(tmp_path / "checkpoints.sqlite3") as manager:
        payload = manager.saver.serde.dumps_typed(_UntrustedCheckpointType())
        restored = manager.saver.serde.loads_typed(payload)

    assert restored == {"value": "must not revive as a Python object"}
    assert not isinstance(restored, _UntrustedCheckpointType)
    manager.close()


class _CounterState(TypedDict):
    count: int


def _counter_graph(saver: SqliteSaver):
    builder = StateGraph(_CounterState)
    builder.add_node("increment", lambda state: {"count": state["count"] + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile(checkpointer=saver)


def test_checkpoint_state_survives_manager_reopen(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "thread-1"}}
    with SqliteCheckpointManager(path) as manager:
        graph = _counter_graph(manager.saver)
        assert graph.invoke({"count": 0}, config)["count"] == 1

    with SqliteCheckpointManager(path) as manager:
        graph = _counter_graph(manager.saver)
        assert graph.get_state(config).values["count"] == 1
