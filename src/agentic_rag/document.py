"""Format-independent content of a document: the blocks every parser produces, the chunks the
chunker makes of them, and those chunks once stored and retrieved, with their document.

The domain types: they depend on nothing else in the package, and everything may import them.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type DivisionKind = Literal[
    "chapter",
    "section",
    "article",
    "paragraph",
    "point",  # a numbered point of a legal text quoted by a guidance document: "3.5.1"
    "recital",
    "annex",
    "heading",  # a heading of a PDF or Markdown document, as printed: "2.1 CASE No. 01: ..."
]


class Division(BaseModel):
    """One level of a document's structure, e.g. Article 33, or its paragraph 1."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: DivisionKind
    number: str | None = None  # as printed in the document: "IV", "33", "2.1"; None if unnumbered
    title: str | None = None


class Block(BaseModel):
    """A piece of content and its place in the document's structure.

    A block is the content of one division, up to the next division or table: its path says
    exactly which article, paragraph or heading the text comes from. Consecutive blocks share a
    path only around a table or a subheading of an EUR-Lex annex. The chunker then merges small
    blocks and splits long ones.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["text", "table"]
    # "text": one line per subparagraph, point or list item (EUR-Lex indents nested points by
    # two spaces; in PDFs, nesting only shows in the bullets: "·", "o", "▪").
    # "table": a Markdown table, header row first.
    text: str = Field(min_length=1)
    path: tuple[Division, ...]  # outermost first: Chapter IV > Section 2 > Article 33 > paragraph 1
    page: int | None = None  # PDFs: the page the block starts on, to cite "p. 14"


class Chunk(BaseModel):
    """What gets embedded, retrieved and given to the LLM: consecutive blocks of one unit.

    Small blocks of a unit are merged up to the token budget, long ones are split between their
    lines. A table is always a chunk of its own, never split.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["text", "table"]
    text: str = Field(min_length=1)  # the blocks' lines, and the headings entered between them
    path: tuple[Division, ...]  # the longest path its blocks share: Article 33 for 33(1) to 33(5)
    page: int | None = None  # PDFs: the page the chunk starts on


class IndexedDocument(BaseModel):
    """A stored document: how to cite it, who may read it, and what its chunks were built from."""

    model_config = ConfigDict(frozen=True)

    id: str  # the manifest's slug: "gdpr"
    title: str
    short_title: str  # what the answer cites: "GDPR"
    source_url: str | None  # None for internal documents
    regulations: tuple[str, ...]
    groups: tuple[str, ...]  # who may read it: ("public",); empty: nobody
    sha256: str  # of the source file: an unchanged file is not ingested again
    embedding_model: str  # of all its chunks: a query is only compared with vectors of its model


class StoredChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: IndexedDocument
    position: int  # in the document, 0-based
    chunk: Chunk

    @property
    def key(self) -> str:
        """How runs and judgments name the chunk: "gdpr#12"."""
        return f"{self.document.id}#{self.position}"


class RetrievedChunk(StoredChunk):
    score: float  # higher is more relevant; only comparable within one retriever
