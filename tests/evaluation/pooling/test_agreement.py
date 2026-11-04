import pytest

from agentic_rag.evaluation.pooling.agreement import (
    MAX_FALSE_REJECTIONS,
    Proportion,
    agreement,
)
from agentic_rag.evaluation.pooling.judge import StatedFact, Verdict
from agentic_rag.evaluation.pooling.review import ReviewEntry
from agentic_rag.evaluation.pooling.selection import PoolJudgment

_FACT = "notify the supervisory authority without undue delay"
_QUOTE = "shall notify the supervisory"
_TEXT = "Article 33. The controller shall notify the supervisory authority without undue delay."


def _entry(verdict: str, fact: object = 0, quote: object = "") -> ReviewEntry:
    return ReviewEntry.model_validate(
        {
            "question_id": "q",
            "chunk": "gdpr#1",
            "question": "?",
            "answer": "!",
            "facts": [_FACT, "the right to lodge a complaint"],
            "location": "GDPR",
            "text": _TEXT,
            "verdict": verdict,
            "fact": fact,
            "quote": quote,
        }
    )


def _reviewed(key: str, verdict: str, fact: int | list[int] = 0) -> ReviewEntry:
    quote: str | list[str] = ""
    if verdict == "relevant":
        quote = _QUOTE if isinstance(fact, int) else [_QUOTE] * len(fact)
    elif verdict == "new_fact":
        quote = "Article 33. The controller shall"
    return _entry(verdict, fact, quote).model_copy(update={"chunk": key})


def _llm(key: str, *facts: int) -> PoolJudgment:
    stated = tuple(StatedFact(fact=f, quote=_QUOTE) for f in facts)
    return PoolJudgment(
        question_id="q", chunk=key, model="m", verdict=Verdict(reason="r", facts=stated)
    )


def test_the_agreement_is_measured_stratum_by_stratum() -> None:
    llm = [
        _llm("gdpr#1", 1),  # confirmed, same fact
        _llm("gdpr#2", 1),  # confirmed, other fact
        _llm("gdpr#3", 2),  # wrongly found relevant
        _llm("gdpr#4"),  # rightly rejected
        _llm("gdpr#5"),  # wrongly rejected
        _llm("gdpr#6"),  # a fact the golden set misses: a right rejection
    ]
    review = [
        _reviewed("gdpr#1", "relevant", 1),
        _reviewed("gdpr#2", "relevant", [1, 2]),
        _reviewed("gdpr#3", "rejected"),
        _reviewed("gdpr#4", "rejected"),
        _reviewed("gdpr#5", "relevant", 2),
        _reviewed("gdpr#6", "new_fact"),
    ]

    result = agreement(llm, review)

    assert (result.confirmed.k, result.confirmed.n) == (2, 3)
    assert (result.false_rejections.k, result.false_rejections.n) == (1, 3)
    assert result.new_facts == 1
    assert (result.same_facts.k, result.same_facts.n) == (1, 2)


def test_the_unchecked_verdicts_do_not_count() -> None:
    llm = [_llm("gdpr#1"), _llm("gdpr#2"), _llm("gdpr#3", 1)]  # only gdpr#1 reviewed

    result = agreement(llm, [_reviewed("gdpr#1", "rejected")])

    assert (result.false_rejections.k, result.false_rejections.n) == (0, 1)
    assert result.confirmed.n == 0


def test_the_rule_keeps_the_rejections_up_to_four_errors_in_a_hundred() -> None:
    llm = [_llm(f"gdpr#{i}") for i in range(100)]

    def kept(errors: int) -> bool:
        review = [_reviewed(f"gdpr#{i}", "relevant", 1) for i in range(errors)]
        review += [_reviewed(f"gdpr#{i}", "rejected") for i in range(errors, 100)]
        return agreement(llm, review).rejections_kept

    assert MAX_FALSE_REJECTIONS == 0.10
    assert kept(4)
    assert not kept(5)


def test_a_proportion_carries_its_interval() -> None:
    proportion = Proportion.of(4, 100)

    assert proportion.low < 0.04 < proportion.high


def test_an_unfinished_review_is_refused() -> None:
    with pytest.raises(ValueError, match="fix the review first: q / gdpr#1: no verdict"):
        agreement([_llm("gdpr#1")], [_reviewed("gdpr#1", "")])


def test_a_review_of_pairs_the_llm_never_judged_is_refused() -> None:
    with pytest.raises(ValueError, match="never judged"):
        agreement([_llm("gdpr#1")], [_reviewed("gdpr#2", "rejected")])
