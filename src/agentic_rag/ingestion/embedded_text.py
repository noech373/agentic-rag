"""What gets embedded for each chunk: its text, possibly after what situates it.

Only the vectors change: the stored text, which BM25, the LLM and the metrics read, stays the
chunk's own.
"""

from collections.abc import Callable, Sequence

from agentic_rag.citation import titled_path
from agentic_rag.document import Chunk
from agentic_rag.ingestion.contextual.base import Contextualizer
from agentic_rag.ingestion.contextual.prompt import render_document

type TextsToEmbed = Callable[[str, Sequence[Chunk]], list[str]]  # document title, its chunks


def chunk_texts(title: str, chunks: Sequence[Chunk]) -> list[str]:
    """The baseline: each chunk's text alone."""
    return [chunk.text for chunk in chunks]


def titled_texts(title: str, chunks: Sequence[Chunk]) -> list[str]:
    """The witness without an LLM: the document title (as the LLM gets it) and the titled path."""
    return [_prefixed(" > ".join(filter(None, [title, titled_path(c)])), c.text) for c in chunks]


class ContextualTexts:
    """Contextual retrieval: what an LLM writes to situate each chunk, then its text."""

    def __init__(self, contextualizer: Contextualizer) -> None:
        self.contextualizer = contextualizer

    def __call__(self, title: str, chunks: Sequence[Chunk]) -> list[str]:
        document = render_document(title, chunks)
        return [_prefixed(self.contextualizer.situate(document, c), c.text) for c in chunks]


def _prefixed(prefix: str, text: str) -> str:
    return f"{prefix}\n\n{text}" if prefix else text
