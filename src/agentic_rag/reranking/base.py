"""Rerankers: a model that scores texts against a question, finer than a first retrieval."""

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One score per text, in the same order: higher is more relevant."""
        ...
