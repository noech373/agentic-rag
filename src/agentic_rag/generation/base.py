"""Generators: from a question and its sources to an answer that cites them.

The RAG depends only on `Generator`: another LLM or provider is one new class.
"""

import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from agentic_rag.storage.base import RetrievedChunk

NO_ANSWER = "I don't know based on the available sources."

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # "[2]", and "[1, 3]" although asked for [1][3]


class GenerationError(Exception):
    """The LLM gave no complete answer."""


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str  # cites the sources by number: "... within 72 hours [1]."
    sources: tuple[RetrievedChunk, ...]  # what the LLM was given: [1] is the first

    @property
    def cited(self) -> dict[int, RetrievedChunk]:
        """The sources the text cites, by number; numbers without a source are ignored."""
        numbers = {int(n) for group in _CITATION.findall(self.text) for n in group.split(",")}
        return {n: self.sources[n - 1] for n in sorted(numbers) if 1 <= n <= len(self.sources)}


class Generator(Protocol):
    def generate(self, question: str, sources: Sequence[RetrievedChunk]) -> Answer: ...
