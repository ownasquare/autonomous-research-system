"""Deterministic, dependency-local embeddings for offline vector recall."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from hashlib import blake2b
from typing import Any

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class DeterministicEmbeddingFunction:
    """Hash normalized tokens into fixed, L2-normalized vectors.

    The function is intentionally local and stable: the offline demo never sends
    report text to an embedding provider, and reopening the database produces the
    same vectors on every supported Python version.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 32:
            raise ValueError("embedding dimensions must be at least 32")
        self.dimensions = dimensions

    def __call__(self, documents: Sequence[str]) -> list[list[float]]:
        return [self._embed(document) for document in documents]

    def _embed(self, document: str) -> list[float]:
        tokens = _TOKEN_PATTERN.findall(document.casefold()) or ["empty"]
        counts = Counter(tokens)
        vector = [0.0] * self.dimensions
        for token, count in counts.items():
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    @staticmethod
    def name() -> str:
        return "research-system-deterministic-v1"

    def get_config(self) -> dict[str, Any]:
        return {"dimensions": self.dimensions}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> DeterministicEmbeddingFunction:
        return DeterministicEmbeddingFunction(dimensions=int(config.get("dimensions", 384)))

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:
        dimensions = config.get("dimensions", 384)
        if not isinstance(dimensions, int) or dimensions < 32:
            raise ValueError("embedding dimensions must be an integer of at least 32")

    @staticmethod
    def validate_config_update(old_config: dict[str, Any], new_config: dict[str, Any]) -> None:
        if old_config.get("dimensions", 384) != new_config.get("dimensions", 384):
            raise ValueError("embedding dimensions cannot change for an existing index")
