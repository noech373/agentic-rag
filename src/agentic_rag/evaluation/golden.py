"""Golden set: questions with the facts their answer needs, each fact as verbatim quotes.

A chunk states a fact when it contains one of its quotes whole, after normalization.
"""

import re
import tomllib
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type Category = Literal["factual", "vague", "table", "multi-document", "unanswerable"]


class Quote(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document: str  # where the quote was taken from; it may be found in other documents too
    quote: str


class Evidence(BaseModel):
    """One fact the answer needs, and the wordings that state it: one of them is enough."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    any_of: tuple[Quote, ...] = Field(min_length=1)


class Question(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    category: Category
    origin: Literal["passage", "blind"]
    holdout: bool = False
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    evidence: tuple[Evidence, ...] = ()


class GoldenSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: tuple[Question, ...] = Field(min_length=1)


def load_golden_set(path: Path) -> GoldenSet:
    with path.open("rb") as f:
        return GoldenSet.model_validate(tomllib.load(f))


_MARKS = str.maketrans({"✔": " yes ", "✓": " yes ", "✘": " no ", "✗": " no "})  # tables' check mark


def normalize(text: str) -> str:
    """Lowercase words only: a quote matches whatever its case, punctuation and spacing.

    A check mark counts as the word it stands for, so that a table row keeps what it says.
    """
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_MARKS)
    return " ".join(re.findall(r"\w+", folded))


def contains(normalized_text: str, quote: str) -> bool:
    """Whether `quote` is in a text already passed through `normalize`, as whole words.

    Whole words: "72 hours" is not in "172 hours".
    """
    return f" {normalize(quote)} " in f" {normalized_text} "
