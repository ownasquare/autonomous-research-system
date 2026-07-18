from research_system.agents.common import (
    MAX_MODEL_CONTEXT_CHARS,
    MAX_MODEL_CONTEXT_TURNS,
    bounded_prior_conversation,
)
from research_system.state import (
    MAX_CONVERSATION_CHARS,
    MAX_CONVERSATION_TURNS,
    MAX_STORED_TURN_CHARS,
    merge_bounded_conversation,
)


def test_checkpoint_conversation_window_is_bounded() -> None:
    turns = [
        {"role": "assistant", "content": f"turn-{index} " + ("x" * 20_000)} for index in range(12)
    ]

    merged = merge_bounded_conversation([], turns)

    assert len(merged) <= MAX_CONVERSATION_TURNS
    assert sum(len(turn["content"]) for turn in merged) <= MAX_CONVERSATION_CHARS
    assert all(len(turn["content"]) <= MAX_STORED_TURN_CHARS for turn in merged)
    assert merged[-1]["content"].startswith("turn-11")


def test_model_context_excludes_current_user_and_has_tighter_bounds() -> None:
    state = {
        "conversation": [
            {"role": "user", "content": f"prior question {index}"}
            if index % 2 == 0
            else {"role": "assistant", "content": f"prior answer {index} " + ("a" * 5_000)}
            for index in range(10)
        ]
        + [{"role": "user", "content": "CURRENT FOLLOW-UP"}]
    }

    context = bounded_prior_conversation(state)

    assert len(context) <= MAX_MODEL_CONTEXT_TURNS
    assert sum(len(turn.content) for turn in context) <= MAX_MODEL_CONTEXT_CHARS
    assert all("CURRENT FOLLOW-UP" not in turn.content for turn in context)
