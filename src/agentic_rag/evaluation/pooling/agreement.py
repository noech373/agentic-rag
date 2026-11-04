"""How far the LLM's verdicts can be trusted, measured on the person's review."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from agentic_rag.evaluation.aggregate import wilson_interval
from agentic_rag.evaluation.pooling.review import ReviewEntry, review_errors
from agentic_rag.evaluation.pooling.selection import PoolJudgment

MAX_FALSE_REJECTIONS = 0.10  # rule fixed before the review: upper bound of the 95 % interval


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
