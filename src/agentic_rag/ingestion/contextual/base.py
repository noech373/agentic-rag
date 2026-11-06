"""Contextualizers: from a chunk and its document to what situates the chunk for retrieval.

The ingestion depends only on `Contextualizer`: another LLM or provider is one new class.
"""

from typing import Protocol

from agentic_rag.document import Chunk


class Contextualizer(Protocol):
    def situate(self, document: str, chunk: Chunk) -> str:
        """The context of a chunk, given its document as `render_document` gives it."""
        ...
