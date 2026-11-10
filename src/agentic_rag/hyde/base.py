"""Hypothetical answers (HyDE): what an LLM answers to a question without any document.

The retrieval searches with that answer: it is worded like the texts, not like the question.
"""

from typing import Protocol


class AnswerWriter(Protocol):
    def write(self, question: str) -> str:
        """A plausible answer to `question`, from the model's own knowledge."""
        ...
