"""What the LLM reads to situate a chunk: the document, with the titles its chunks leave out."""

from collections.abc import Sequence

from agentic_rag.citation import titled_path
from agentic_rag.document import Chunk

INSTRUCTIONS = """\
You situate a passage within its document, to improve search retrieval of the passage.
Answer only with a succinct context of one or two sentences, in English, without repeating the \
passage.
State which part of the document the passage belongs to and what it is about.
Do not summarize the rest of the document; mention another provision only if it is needed to \
understand this passage.
Start by naming the document (e.g. "the GDPR", "ENISA's NIS2 implementation guidance").
"""


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


def build_input(document: str, chunk: Chunk) -> str:
    """The rendered document, then the chunk to situate: its location tells repeated texts apart."""
    location = f"<location>{titled_path(chunk)}</location>\n" if chunk.path else ""
    return f"{document}\n\n<passage>\n{location}{chunk.text}\n</passage>"
