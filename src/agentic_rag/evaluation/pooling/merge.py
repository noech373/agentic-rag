"""Merge a finished pooling review into golden.toml and judgments.toml, keeping their comments."""

import re
import tomllib
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from pydantic import TypeAdapter

from agentic_rag.evaluation.golden import Question, Quote, contains, normalize
from agentic_rag.evaluation.judgments import Judgments
from agentic_rag.evaluation.pooling.agreement import agreement
from agentic_rag.evaluation.pooling.review import ReviewEntry, toml_string
from agentic_rag.evaluation.pooling.selection import PoolJudgment

REJECTION_WORDS = 12  # enough to tell a chunk from the others, short enough to read
_WORD = re.compile(r"\w+|[✔✓✘✗]")  # a check mark is a word for `normalize`


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


def rejection_candidates(
    llm: Sequence[PoolJudgment], review: Sequence[ReviewEntry]
) -> list[tuple[str, str]]:
    """The (question, chunk) pairs that may be rejected, in the order of the LLM's verdicts.

    Candidates only: `stating_a_fact` then takes out those that state a fact of their question.
    The person's verdict comes first. The LLM's rejections nobody checked count only if the rule
    keeps them. A "new_fact" is no rejection: the chunk waits for the fact to be added by hand.
    """
    reviewed = {(entry.question_id, entry.chunk): entry.verdict for entry in review}
    keep_unchecked = agreement(llm, review).rejections_kept
    pairs: list[tuple[str, str]] = []
    for judgment in llm:
        pair = (judgment.question_id, judgment.chunk)
        verdict = reviewed.get(pair)
        if verdict == "rejected" or (
            verdict is None and keep_unchecked and not judgment.verdict.relevant
        ):
            pairs.append(pair)
    return pairs


def stating_a_fact(
    pairs: Sequence[tuple[str, str]], questions: Mapping[str, Question], texts: Mapping[str, str]
) -> list[tuple[str, str]]:
    """The pairs whose chunk contains a wording of its question: they count as useful anyway.

    A wording from a relevant chunk may also be in a chunk rejected unread.
    """
    return [
        (question_id, key)
        for question_id, key in pairs
        if any(
            contains(normalize(texts[key]), quote.quote)
            for evidence in questions[question_id].evidence
            for quote in evidence.any_of
        )
    ]


def rejection_quotes(pairs: Sequence[tuple[str, str]], texts: Mapping[str, str]) -> Judgments:
    """For each question, a quote of each of its rejected chunks; `texts`: every chunk, by key."""
    normalized = [normalize(text) for text in texts.values()]
    quotes = {key: _own_quote(key, texts[key], normalized) for key in {key for _, key in pairs}}
    judgments: dict[str, tuple[Quote, ...]] = {}
    for question_id, key in pairs:
        judgments[question_id] = (*judgments.get(question_id, ()), quotes[key])
    return judgments


def _own_quote(key: str, text: str, normalized: Sequence[str]) -> Quote:
    """The first `REJECTION_WORDS` words of the chunk that are in no other chunk.

    A shared quote, such as a table's header, would reject the other chunks too, unread.
    Exact copies of the chunk do not count: the same text gets the same verdict.
    """
    own = normalize(text)
    others = [other for other in normalized if other != own]
    words = list(_WORD.finditer(text))
    for first in range(max(len(words) - REJECTION_WORDS, 0) + 1 if words else 0):
        last = words[min(first + REJECTION_WORDS, len(words)) - 1]
        quote = " ".join(text[words[first].start() : last.end()].split())
        if not any(contains(other, quote) for other in others):
            return Quote(document=key.rpartition("#")[0], quote=quote)
    raise ValueError(f"{key}: no {REJECTION_WORDS} words of it are in this chunk alone")


def add_rejections(judgments: str, new: Judgments) -> str:
    """The text of judgments.toml with the new quotes; its leading comments are kept as they are.

    One it already has is not added again. Questions are sorted, so that the file diffs well.
    """
    existing = TypeAdapter(Judgments).validate_python(tomllib.loads(judgments))
    merged = dict(existing)
    for question_id, quotes in new.items():
        for quote in quotes:
            held = merged.get(question_id, ())
            if normalize(quote.quote) not in {normalize(q.quote) for q in held}:
                merged[question_id] = (*held, quote)

    comments: list[str] = []
    for line in judgments.splitlines():
        if line and not line.startswith("#"):
            break
        comments.append(line)
    blocks = ["\n".join(comments).rstrip()]
    for question_id in sorted(merged):
        lines = "".join(
            f"  {{ document = {toml_string(q.document)}, quote = {toml_string(q.quote)} }},\n"
            for q in merged[question_id]
        )
        blocks.append(f"{question_id} = [\n{lines}]")
    return "\n\n".join(blocks) + "\n"
