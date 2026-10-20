"""Format-independent content of a parsed document: what every parser produces for the chunker."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type DivisionKind = Literal["chapter", "section", "article", "paragraph", "recital", "annex"]


class Division(BaseModel):
    """One level of a document's structure, e.g. Article 33, or its paragraph 1."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: DivisionKind
    number: str  # as printed in the document: "IV", "33", "1"
    title: str | None = None


class Block(BaseModel):
    """A piece of content and its place in the document's structure.

    A block never spans two divisions, so its path says exactly which article, paragraph or
    annex the text comes from. The chunker then merges small blocks and splits long ones.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["text", "table"]
    # "text": one line per subparagraph or point, nested points indented by two spaces.
    # "table": a Markdown table, header row first.
    text: str = Field(min_length=1)
    path: tuple[Division, ...]  # outermost first: Chapter IV > Section 2 > Article 33 > paragraph 1
