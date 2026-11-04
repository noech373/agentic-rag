"""Pooling: judge the chunks that several runs rank high and that no judgment covers yet.

The pool of a question is the union of the top `POOL_DEPTH` of each run. A chunk is judged for
one question: the same chunk may be pooled, and judged, for several questions. An LLM pre-judges
the pairs, each verdict appended to `llm.jsonl` as it comes, so that an interrupted pooling resumes.
A person then checks, blind, all those it found relevant and a random sample of the others
(`review.toml`).
"""

import json
import random
import tomllib
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

from agentic_rag.citation import citation
from agentic_rag.document import StoredChunk
from agentic_rag.evaluation.aggregate import wilson_interval
from agentic_rag.evaluation.comparison import IncomparableRunsError
from agentic_rag.evaluation.golden import Question, contains, normalize
from agentic_rag.evaluation.judgments import Judgments
from agentic_rag.evaluation.metrics import is_judged
from agentic_rag.evaluation.record import RunRecord
from agentic_rag.evaluation.relevance_judge import RelevanceJudge, StatedFact, Verdict
from agentic_rag.evaluation.runner import JUDGED_K

POOLING_DIR = Path("eval/pooling")  # llm.jsonl and review.toml
POOL_DEPTH = JUDGED_K  # the top that judged@k checks
REVIEW_SAMPLE = 100  # rejections checked by hand: at most 4 misses keep the upper bound under 10 %
QUOTE_WORDS = (4, 25)  # a quote's length, as asked of the LLM
MAX_FALSE_REJECTIONS = 0.10  # rule fixed before the review: upper bound of the 95 % interval


class PoolJudgment(BaseModel):
    """One line of llm.jsonl: the LLM's verdict on one chunk for one question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    chunk: str  # "gdpr#12"
    model: str
    verdict: Verdict


def pool(question_id: str, runs: Sequence[RunRecord], depth: int = POOL_DEPTH) -> list[str]:
    """The first `depth` chunks of each run for the question, each once, in the order first seen."""
    pooled: dict[str, None] = {}  # a dict keeps the insertion order
    for run in runs:
        for result in run.results:
            if result.question_id == question_id:
                pooled.update(dict.fromkeys(result.retrieved[:depth]))
    return list(pooled)


def chunks_to_judge(
    questions: Sequence[Question],
    runs: Sequence[RunRecord],
    judgments: Judgments,
    chunks: Mapping[str, StoredChunk],
) -> list[tuple[Question, StoredChunk]]:
    """The pooled (question, chunk) pairs that neither the golden set nor the judgments cover.

    Questions absent from the runs (the holdout) have an empty pool.
    """
    if len({run.golden_sha256 for run in runs}) > 1:
        raise IncomparableRunsError("the runs used different golden sets: rerun them")

    to_judge: list[tuple[Question, StoredChunk]] = []
    for question in questions:
        rejected = judgments.get(question.id, ())
        for key in pool(question.id, runs):
            if key not in chunks:
                raise ValueError(f"{key} is not in the index: it changed since the runs")
            chunk = chunks[key]
            if not is_judged(question, rejected, chunk.chunk):
                to_judge.append((question, chunk))
    return to_judge


def not_yet_judged(
    to_judge: Sequence[tuple[Question, StoredChunk]], done: Sequence[PoolJudgment]
) -> list[tuple[Question, StoredChunk]]:
    """The pairs the LLM has no verdict for yet: what an interrupted pooling still has to do."""
    judged = {(j.question_id, j.chunk) for j in done}
    return [(q, c) for q, c in to_judge if (q.id, c.key) not in judged]


def judge_pool(
    to_judge: Sequence[tuple[Question, StoredChunk]],
    judge: RelevanceJudge,
    path: Path,
    *,
    workers: int = 8,
) -> list[PoolJudgment]:
    """Judge the pairs in parallel, appending each verdict to `path` as soon as it comes.

    If a call fails, the verdicts already received stay in the file and the rest is cancelled:
    running again judges only what is missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    judged: list[PoolJudgment] = []
    with ThreadPoolExecutor(workers) as executor, path.open("a", encoding="utf-8") as file:
        futures = {executor.submit(judge.judge, q, c): (q, c) for q, c in to_judge}
        try:
            for future in as_completed(futures):  # only this thread writes to the file
                question, chunk = futures[future]
                judgment = PoolJudgment(
                    question_id=question.id,
                    chunk=chunk.key,
                    model=judge.model,
                    verdict=future.result(),
                )
                file.write(judgment.model_dump_json() + "\n")
                file.flush()
                judged.append(judgment)
        except BaseException:  # also Ctrl+C
            executor.shutdown(cancel_futures=True)
            raise
    return judged


def load_pool_judgments(path: Path) -> list[PoolJudgment]:
    """The verdicts saved so far; none before the first pooling."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PoolJudgment.model_validate_json(line) for line in lines if line.strip()]


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


class Proportion(BaseModel):
    """`k` out of `n`, with its 95 % Wilson interval."""

    model_config = ConfigDict(frozen=True)

    k: int
    n: int
    low: float
    high: float

    @classmethod
    def of(cls, k: int, n: int) -> Proportion:
        low, high = wilson_interval(k, n)
        return cls(k=k, n=n, low=low, high=high)


class Agreement(BaseModel):
    """How the LLM's verdicts compare with the person's, stratum by stratum.

    The review holds all the LLM's relevant verdicts but only a sample of its rejections, so a
    single agreement rate over the review would say nothing of the whole pool.
    """

    model_config = ConfigDict(frozen=True)

    confirmed: Proportion  # LLM relevant that the person found relevant too
    false_rejections: Proportion  # LLM rejections that the person found relevant
    new_facts: int  # facts the golden set misses, whatever the LLM said
    same_facts: Proportion  # both relevant: the same facts found

    @property
    def rejections_kept(self) -> bool:
        """Whether the LLM's unchecked rejections may count as judgments (the rule)."""
        return self.false_rejections.high <= MAX_FALSE_REJECTIONS


def agreement(llm: Sequence[PoolJudgment], review: Sequence[ReviewEntry]) -> Agreement:
    """Compare each reviewed pair with the LLM's verdict on it; a new fact counts as rejected."""
    if errors := review_errors(review):
        raise ValueError(f"fix the review first: {errors[0]} ({len(errors)} errors)")
    verdicts = {(j.question_id, j.chunk): j.verdict for j in llm}
    if missing := [
        (e.question_id, e.chunk) for e in review if (e.question_id, e.chunk) not in verdicts
    ]:
        raise ValueError(
            f"reviewed pairs the LLM never judged: {missing[0]} and {len(missing) - 1} more"
        )

    pairs = [(verdicts[(e.question_id, e.chunk)], e) for e in review]
    relevant = [(v, e) for v, e in pairs if v.relevant]
    rejected = [e for v, e in pairs if not v.relevant]
    both = [(v, e) for v, e in relevant if e.verdict == "relevant"]
    return Agreement(
        confirmed=Proportion.of(len(both), len(relevant)),
        false_rejections=Proportion.of(
            sum(e.verdict == "relevant" for e in rejected), len(rejected)
        ),
        new_facts=sum(e.verdict == "new_fact" for e in review),
        same_facts=Proportion.of(
            sum({f.fact for f in v.facts} == {f.fact for f in e.stated} for v, e in both), len(both)
        ),
    )
