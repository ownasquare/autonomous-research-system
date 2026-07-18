from __future__ import annotations

import math

from research_system.memory.embeddings import DeterministicEmbeddingFunction


def test_embeddings_are_deterministic_and_normalized() -> None:
    embeddings = DeterministicEmbeddingFunction(dimensions=64)

    first, second = embeddings(["LangGraph supervisor worker orchestration"] * 2)

    assert first == second
    assert len(first) == 64
    assert math.isclose(
        math.sqrt(sum(value * value for value in first)),
        1.0,
        abs_tol=1e-6,
    )


def test_embeddings_separate_distinct_token_sets() -> None:
    embeddings = DeterministicEmbeddingFunction(dimensions=64)

    graph, biology = embeddings(["agent graph orchestration", "marine biology coral reef"])

    assert graph != biology


def test_embedding_configuration_round_trips() -> None:
    embeddings = DeterministicEmbeddingFunction(dimensions=96)

    rebuilt = DeterministicEmbeddingFunction.build_from_config(embeddings.get_config())

    assert rebuilt.dimensions == 96
    assert rebuilt.name() == "research-system-deterministic-v1"
