"""Merge a finished pooling review into the golden set, keeping the file's comments and layout."""

import re
from collections.abc import Sequence
from typing import NamedTuple

from agentic_rag.evaluation.golden import normalize
from agentic_rag.evaluation.pooling import ReviewEntry, toml_string


class NewWording(NamedTuple):
    """A quote to add to one fact of a golden question."""

    question_id: str
    fact: int  # its number in the question, from 1
    document: str
    quote: str


def new_wordings(review: Sequence[ReviewEntry]) -> list[NewWording]:
    """The quotes of the chunks the person found relevant, on one line.

    A "new_fact" is not among them: adding a fact to a question is decided by hand.
    """
    return [
        NewWording(
            entry.question_id,
            stated.fact,
            entry.chunk.rpartition("#")[0],
            " ".join(stated.quote.split()),
        )
        for entry in review
        if entry.verdict == "relevant"
        for stated in entry.stated
    ]


def add_wordings(golden: str, wordings: Sequence[NewWording]) -> str:
    """The text of golden.toml with the wordings added; one it already has is not added again."""
    for wording in wordings:
        golden = _add_wording(golden, wording)
    return golden


_ANY_OF = re.compile(r"^any_of = \[\n(.*?)^\]$", re.MULTILINE | re.DOTALL)


def _add_wording(golden: str, wording: NewWording) -> str:
    start = golden.find(f"\nid = {toml_string(wording.question_id)}\n")
    if start == -1:
        raise ValueError(f"{wording.question_id} is not in the golden set")
    next_question = golden.find("\n[[questions]]", start)
    end = len(golden) if next_question == -1 else next_question

    facts = list(_ANY_OF.finditer(golden, start, end))
    if not 1 <= wording.fact <= len(facts):
        raise ValueError(f"{wording.question_id} has no fact {wording.fact}")
    fact = facts[wording.fact - 1]
    if normalize(wording.quote) in {
        normalize(quote) for quote in re.findall(r'quote = "((?:[^"\\]|\\.)*)"', fact[1])
    }:
        return golden

    line = (
        f"  {{ document = {toml_string(wording.document)}, "
        f"quote = {toml_string(wording.quote)} }},\n"
    )
    return golden[: fact.end(1)] + line + golden[fact.end(1) :]
