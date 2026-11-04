"""The review: a person checks, blind, all the chunks the LLM found relevant and a random sample
of the others, in `review.toml`.
"""

import json
import random
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

from agentic_rag.citation import citation
from agentic_rag.document import StoredChunk
from agentic_rag.evaluation.golden import Question, contains, normalize
from agentic_rag.evaluation.pooling.judge import StatedFact
from agentic_rag.evaluation.pooling.selection import PoolJudgment

REVIEW_SAMPLE = 100  # rejections checked by hand: at most 4 misses keep the upper bound under 10 %
QUOTE_WORDS = (4, 25)  # a quote's length, as asked of the LLM


def review_sample(
    judgments: Sequence[PoolJudgment], *, sample: int = REVIEW_SAMPLE, seed: int = 0
) -> list[PoolJudgment]:
    """All the LLM's relevant verdicts and `sample` of its rejections drawn at random.

    Sorted by question then chunk, so that the order does not tell which the LLM found relevant.
    """
    ordered = sorted(judgments, key=lambda j: (j.question_id, j.chunk))
    relevant = [j for j in ordered if j.verdict.relevant]
    rejected = [j for j in ordered if not j.verdict.relevant]
    rng = random.Random(seed)  # noqa: S311  (reproducible sample, not security)
    drawn = rng.sample(rejected, min(sample, len(rejected)))
    return sorted([*relevant, *drawn], key=lambda j: (j.question_id, j.chunk))


_REVIEW_HEADER = """\
# Pooling review: does each chunk state one of the facts its question needs? Judge without
# looking at the LLM's verdicts (eval/pooling/llm.jsonl): the agreement is measured on this file.
#
# Set `verdict` to:
# - "relevant" if the chunk, read alone, states one of the `facts` (in any words; a recital
#   counts): set `fact` to its number in `facts` (1 for the first) and `quote` to the words of
#   the chunk that state it, copied exactly (4 to 25 words); if it states several facts, give
#   lists in the same order: `fact = [1, 3]` and `quote = ["…", "…"]`;
# - "new_fact" if it states something a good answer needs that no fact lists (the question
#   misses a fact): keep `fact = 0` and put its words in `quote`;
# - "rejected" otherwise (same topic only, a reference without the fact, or a mere extra).
"""


def write_review(
    entries: Sequence[PoolJudgment],
    questions: Mapping[str, Question],
    chunks: Mapping[str, StoredChunk],
    path: Path,
) -> None:
    """The entries to check by hand, as TOML, without the LLM's verdict; never over a review."""
    blocks = [_REVIEW_HEADER]
    for entry in entries:
        question = questions[entry.question_id]
        chunk = chunks[entry.chunk]
        facts = [
            " | ".join(quote.quote for quote in evidence.any_of) for evidence in question.evidence
        ]
        blocks.append(
            "\n".join(
                [
                    "[[chunks]]",
                    f"question_id = {toml_string(question.id)}",
                    f"chunk = {toml_string(chunk.key)}",
                    f"question = {toml_string(question.question)}",
                    f"answer = {toml_string(question.answer)}",
                    "facts = [\n" + "".join(f"  {toml_string(f)},\n" for f in facts) + "]",
                    f"location = {toml_string(citation(chunk))}",
                    f"text = {_multiline(chunk.chunk.text)}",
                    'verdict = ""',
                    "fact = 0",
                    'quote = ""',
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as file:  # "x": fails if the file exists
            file.write("\n\n".join(blocks) + "\n")
    except FileExistsError:
        raise FileExistsError(f"{path} exists: it may hold a review, rename or delete it") from None


def toml_string(value: str) -> str:
    """A TOML basic string: JSON's escapes are valid TOML ones (DEL is the one TOML adds)."""
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def _multiline(value: str) -> str:
    """A TOML literal string, kept as written for reading, unless it cannot hold the text."""
    controls = any((c < " " and c not in "\t\n") or c == "\x7f" for c in value)
    if "'''" in value or controls:
        return toml_string(value)
    return f"'''\n{value}'''"


class ReviewEntry(BaseModel):
    """One `[[chunks]]` of review.toml, as the person filled it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    chunk: str
    question: str
    answer: str
    facts: tuple[str, ...]
    location: str
    text: str
    verdict: Literal["", "relevant", "new_fact", "rejected"]  # "": not judged yet
    fact: int | tuple[int, ...]  # 0: none; a tuple when the chunk states several facts
    quote: str | tuple[str, ...]

    @property
    def stated(self) -> tuple[StatedFact, ...]:
        """The facts the person found in the chunk, each with its quote; empty if none."""
        facts = self.fact if isinstance(self.fact, tuple) else (self.fact,)
        quotes = self.quote if isinstance(self.quote, tuple) else (self.quote,)
        return tuple(StatedFact(fact=f, quote=q) for f, q in zip(facts, quotes, strict=True) if f)


def load_review(path: Path) -> list[ReviewEntry]:
    with path.open("rb") as f:
        return TypeAdapter(list[ReviewEntry]).validate_python(tomllib.load(f)["chunks"])


def review_errors(entries: Sequence[ReviewEntry]) -> list[str]:
    """Every value the person must fix before the review is used, one message each."""
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        where = f"{entry.question_id} / {entry.chunk}"
        if (entry.question_id, entry.chunk) in seen:
            errors.append(f"{where}: reviewed twice")
        seen.add((entry.question_id, entry.chunk))
        errors.extend(f"{where}: {error}" for error in _entry_errors(entry))
    return errors


def _entry_errors(entry: ReviewEntry) -> list[str]:
    facts = entry.fact if isinstance(entry.fact, tuple) else (entry.fact,)
    quotes = entry.quote if isinstance(entry.quote, tuple) else (entry.quote,)
    if entry.verdict == "":
        return ["no verdict"]
    if len(facts) != len(quotes):
        return [f"{len(facts)} facts but {len(quotes)} quotes"]
    if entry.verdict == "rejected":
        return [] if facts == (0,) and quotes == ("",) else ["a rejection has no fact nor quote"]
    if entry.verdict == "new_fact":
        errors = [] if facts == (0,) else ["a new fact keeps fact = 0"]
        return errors + _quote_errors(entry.text, quotes[0])
    errors = [] if len(set(facts)) == len(facts) else ["a fact is given twice"]
    for fact, quote in zip(facts, quotes, strict=True):
        if not 1 <= fact <= len(entry.facts):
            errors.append(f"fact {fact}: the question has facts 1 to {len(entry.facts)}")
        errors += [f"fact {fact}: {error}" for error in _quote_errors(entry.text, quote)]
    return errors


def _quote_errors(text: str, quote: str) -> list[str]:
    """A quote must be in the chunk (case, punctuation and spacing aside) and of 4 to 25 words."""
    words = len(normalize(quote).split())
    low, high = QUOTE_WORDS
    errors = [] if low <= words <= high else [f"the quote has {words} words, not {low} to {high}"]
    if words and not contains(normalize(text), quote):
        errors.append("the quote is not in the chunk")
    return errors
