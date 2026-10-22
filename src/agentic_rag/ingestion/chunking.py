"""Structural chunking: merge the small blocks of a document and split the long ones."""

import functools
from collections.abc import Callable, Sequence

import tiktoken

from agentic_rag.document import Block, Chunk, Division, DivisionKind

type TokenCounter = Callable[[str], int]
type _Path = tuple[Division, ...]

# A chunk never spans two of these. The unit of a block is its path down to the first of them
# (Article 33, Recital 85, Annex III, point 3.2.4), or its whole path if it has none (headings).
_UNITS: frozenset[DivisionKind] = frozenset({"article", "recital", "annex", "point"})


@functools.cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")  # the tokenizer of text-embedding-3-small, -large


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def chunk_blocks(
    blocks: Sequence[Block], *, max_tokens: int = 512, counter: TokenCounter = count_tokens
) -> list[Chunk]:
    """Merge consecutive blocks of the same unit up to `max_tokens`, split the longer ones.

    The budget is checked on the joined text: token counts do not add up ("." and "\\n" make one
    token). A single line over the budget stays whole: the longest in the corpus (a recital of
    833 tokens) is far below the 8191 tokens the embedding model accepts.
    """
    chunks: list[Chunk] = []
    previous: _Path = ()  # the path of the last block added
    for block in blocks:
        for piece in _pieces(block, max_tokens, counter):
            merged = _merge(chunks[-1], previous, block, piece) if chunks else None
            if merged is not None and counter(merged.text) <= max_tokens:
                chunks[-1] = merged
            else:
                chunks.append(Chunk(kind=block.kind, text=piece, path=block.path, page=block.page))
            previous = block.path
    return chunks


def _pieces(block: Block, max_tokens: int, counter: TokenCounter) -> list[str]:
    """The block whole if it fits or is a table, else its lines, each with its nested lines."""
    if block.kind == "table" or counter(block.text) <= max_tokens:
        return [block.text]
    pieces: list[str] = []
    for line in block.text.split("\n"):
        if pieces and line.startswith(" "):  # a nested point stays with the line introducing it
            pieces[-1] += f"\n{line}"
        else:
            pieces.append(line)
    return pieces  # chunk_blocks merges them back up to the budget


def _merge(chunk: Chunk, previous: _Path, block: Block, piece: str) -> Chunk | None:
    """The chunk with the piece of the block appended, or None if they must stay apart."""
    if "table" in (chunk.kind, block.kind) or _unit(chunk.path) != _unit(block.path):
        return None
    # The merged chunk keeps only the shared path, so the headings it drops ("GUIDANCE" under
    # point 3.2.4 of the ENISA guidance) become lines, or the text would lose them: those of the
    # chunk go first, those the block enters go before its piece.
    path = _common_prefix(chunk.path, block.path)
    left = _headings(chunk.path[len(path) :])
    entered = _headings(block.path[len(_common_prefix(previous, block.path)) :])
    text = "\n".join([*left, chunk.text, *entered, piece])
    return chunk.model_copy(update={"text": text, "path": path})


def _headings(divisions: _Path) -> list[str]:
    return [d.title for d in divisions if d.kind == "heading" and d.title]


def _unit(path: _Path) -> _Path:
    for i, division in enumerate(path):
        if division.kind in _UNITS:
            return path[: i + 1]
    return path


def _common_prefix(a: _Path, b: _Path) -> _Path:
    length = 0
    while length < min(len(a), len(b)) and a[length] == b[length]:
        length += 1
    return a[:length]
