"""A writer that never asks twice for the answer to the same question, across runs.

The runs of one measure (before and after pooling) then search with the same answers: an LLM
alone may word them differently each time, and bring up chunks nobody judged.
"""

import json
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentic_rag.hyde.base import AnswerWriter
from agentic_rag.hyde.prompt import INSTRUCTIONS

ANSWERS_PATH = Path("eval/hyde.jsonl")  # committed: the runs can be redone the same


class CachedAnswer(BaseModel):
    """One line of the file. Only `key` and `answer` are used, the rest is there to read it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    model: str
    question: str
    answer: str


class CachedAnswerWriter(AnswerWriter):
    def __init__(self, writer: AnswerWriter, path: Path) -> None:
        self._writer = writer
        self._path = path
        self._answers = _load(path)

    @property
    def name(self) -> str:
        return self._writer.name

    def write(self, question: str) -> str:
        key = _key(self.name, question)
        if key in self._answers:
            return self._answers[key]
        answer = self._writer.write(question)
        line = CachedAnswer(key=key, model=self.name, question=question, answer=answer)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(line.model_dump_json() + "\n")
        self._answers[key] = answer
        return answer


def _key(model: str, question: str) -> str:
    """The fingerprint of the request: any change to it asks the LLM again."""
    request = json.dumps([model, INSTRUCTIONS, question])
    return sha256(request.encode()).hexdigest()


def _load(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        lines = (CachedAnswer.model_validate_json(line) for line in file)
        return {line.key: line.answer for line in lines}
