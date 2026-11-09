"""What gets embedded for each chunk: its text, possibly after what situates it.

Only the vectors change: the stored text, which BM25, the LLM and the metrics read, stays the
chunk's own.
"""

from collections.abc import Callable, Sequence

from agentic_rag.citation import titled_path
from agentic_rag.document import Chunk

type TextsToEmbed = Callable[[str, Sequence[Chunk]], list[str]]  # document title, its chunks


def chunk_texts(title: str, chunks: Sequence[Chunk]) -> list[str]:
    """The baseline: each chunk's text alone."""
    return [chunk.text for chunk in chunks]


def titled_texts(title: str, chunks: Sequence[Chunk]) -> list[str]:
    """Each chunk's titled path, then its text: what the titles alone bring, without an LLM."""
    return [_prefixed(titled_path(chunk), chunk.text) for chunk in chunks]


def _prefixed(prefix: str, text: str) -> str:
    return f"{prefix}\n\n{text}" if prefix else text
