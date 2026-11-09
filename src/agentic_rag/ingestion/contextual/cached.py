"""A contextualizer that never asks twice for the same context, across ingestions.

Each context is appended to a JSONL file as soon as it comes: an interrupted ingestion resumes,
and ingesting again costs nothing and gives the same texts, which an LLM alone would not.
"""

import json
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentic_rag.citation import titled_path
from agentic_rag.document import Chunk
from agentic_rag.ingestion.contextual.base import Contextualizer
from agentic_rag.ingestion.contextual.prompt import INSTRUCTIONS, build_input

CONTEXTS_PATH = Path("eval/contexts.jsonl")  # committed: ingesting again costs nothing
_PASSAGE_CHARS = 80


class CachedContext(BaseModel):
    """One line of the file. Only `key` and `context` are used, the rest is there to read it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    model: str
    location: str
    passage: str  # how the chunk starts
    context: str


class CachedContextualizer(Contextualizer):
    def __init__(self, contextualizer: Contextualizer, path: Path) -> None:
        self._contextualizer = contextualizer
        self._path = path
        self._contexts = _load(path)

    @property
    def name(self) -> str:
        return self._contextualizer.name

    def situate(self, document: str, chunk: Chunk) -> str:
        key = _key(self.name, document, chunk)
        if key in self._contexts:
            return self._contexts[key]
        context = self._contextualizer.situate(document, chunk)
        line = CachedContext(
            key=key,
            model=self.name,
            location=titled_path(chunk),
            passage=chunk.text[:_PASSAGE_CHARS],
            context=context,
        )
        with self._path.open("a", encoding="utf-8") as file:
            file.write(line.model_dump_json() + "\n")
        self._contexts[key] = context
        return context


def _key(model: str, document: str, chunk: Chunk) -> str:
    """The fingerprint of the request: any change to it asks the LLM again."""
    request = json.dumps([model, INSTRUCTIONS, build_input(document, chunk)])
    return sha256(request.encode()).hexdigest()


def _load(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        lines = (CachedContext.model_validate_json(line) for line in file)
        return {line.key: line.context for line in lines}
