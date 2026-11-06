"""What the LLM reads to situate a chunk: the document, with the titles its chunks leave out."""

from collections.abc import Sequence

from agentic_rag.citation import titled_path
from agentic_rag.document import Chunk


def render_document(title: str, chunks: Sequence[Chunk]) -> str:
    """The chunks in order, each path once: consecutive chunks often share it."""
    parts = [f"<title>{title}</title>"]
    previous = None
    for chunk in chunks:
        if chunk.path and chunk.path != previous:
            parts.append(f"## {titled_path(chunk)}")
        previous = chunk.path
        parts.append(chunk.text)
    return "<document>\n" + "\n\n".join(parts) + "\n</document>"
