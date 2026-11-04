from statistics import fmean

import pytest

from agentic_rag.evaluation.aggregate import (
    bootstrap_ci,
    mean_scores,
    mean_scores_by_category,
    paired_differences,
    wilson_interval,
)
from agentic_rag.evaluation.golden import Category
from agentic_rag.evaluation.runner import QuestionResult


def _result(
    recall_at_8: float,
    mrr: float,
    id: str = "q",
    category: Category = "factual",
    judged: float = 1.0,
) -> QuestionResult:
    return QuestionResult(
        question_id=id,
        category=category,
        recall={"recall@8": recall_at_8},
        mrr=mrr,
        judged=judged,
        retrieved=(),
    )


def test_the_mean_is_taken_per_metric() -> None:
    results = [_result(1.0, 1.0, "a", judged=1.0), _result(0.5, 0.0, "b", judged=0.5)]

    assert mean_scores(results) == {"recall@8": 0.75, "mrr@8": 0.5, "judged@8": 0.75}


def test_every_question_weighs_the_same_whatever_its_number_of_facts() -> None:
    # 1 of 1 fact and 1 of 4 facts: the mean of the two recalls, not 2 facts out of 5
    results = [_result(1.0, 1.0, "a"), _result(0.25, 1.0, "b")]

    assert mean_scores(results)["recall@8"] == pytest.approx(0.625)


def test_no_results_have_no_mean() -> None:
    with pytest.raises(ValueError, match="no results"):
        mean_scores([])


def test_the_means_are_taken_within_each_category() -> None:
    results = [
        _result(1.0, 1.0, "a"),
        _result(0.5, 0.5, "b"),
        _result(0.0, 0.0, "c", category="table"),
    ]

    assert mean_scores_by_category(results) == {
        "factual": {"recall@8": 0.75, "mrr@8": 0.75, "judged@8": 1.0},
        "table": {"recall@8": 0.0, "mrr@8": 0.0, "judged@8": 1.0},
    }


def test_a_category_without_results_is_absent() -> None:
    assert mean_scores_by_category([_result(1.0, 1.0)]).keys() == {"factual"}


def test_no_results_have_no_category_means() -> None:
    assert mean_scores_by_category([]) == {}


_SCORES = [1.0, 1.0, 1.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.5, 1.0]


def test_the_interval_of_identical_scores_is_that_score() -> None:
    assert bootstrap_ci([0.5] * 8) == (0.5, 0.5)


def test_the_interval_surrounds_the_mean() -> None:
    low, high = bootstrap_ci(_SCORES)

    assert low <= fmean(_SCORES) <= high
    assert low < high


def test_the_same_seed_gives_the_same_interval() -> None:
    assert bootstrap_ci(_SCORES, seed=1) == bootstrap_ci(_SCORES, seed=1)


def test_a_higher_confidence_gives_a_wider_interval() -> None:
    low_90, high_90 = bootstrap_ci(_SCORES, confidence=0.90)
    low_99, high_99 = bootstrap_ci(_SCORES, confidence=0.99)

    assert low_99 <= low_90
    assert high_90 <= high_99


def test_more_questions_give_a_narrower_interval() -> None:
    low, high = bootstrap_ci(_SCORES)
    low_x10, high_x10 = bootstrap_ci(_SCORES * 10)

    assert high_x10 - low_x10 < high - low


def test_no_scores_have_no_interval() -> None:
    with pytest.raises(ValueError, match="no scores"):
        bootstrap_ci([])


def test_the_difference_is_taken_question_by_question() -> None:
    baseline = [_result(0.5, 0.0, "a"), _result(1.0, 1.0, "b")]
    candidate = [_result(1.0, 0.5, "a"), _result(0.5, 1.0, "b")]

    assert paired_differences(baseline, candidate, "recall@8") == [0.5, -0.5]
    assert paired_differences(baseline, candidate, "mrr@8") == [0.5, 0.0]


def test_questions_are_paired_by_id_not_by_position() -> None:
    baseline = [_result(0.5, 0.0, "a"), _result(1.0, 1.0, "b")]
    candidate = [_result(0.5, 1.0, "b"), _result(1.0, 0.5, "a")]

    assert paired_differences(baseline, candidate, "recall@8") == [0.5, -0.5]


def test_runs_on_different_questions_cannot_be_paired() -> None:
    with pytest.raises(ValueError, match="same questions"):
        paired_differences([_result(1.0, 1.0, "a")], [_result(1.0, 1.0, "b")], "mrr@8")


def test_a_run_compared_with_itself_has_no_difference_at_all() -> None:
    run = [_result(0.5, 0.25, "a"), _result(1.0, 1.0, "b")]

    differences = paired_differences(run, run, "mrr@8")

    assert differences == [0.0, 0.0]
    assert bootstrap_ci(differences) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("k", "n", "high"),
    [(0, 100, 0.037), (2, 100, 0.070), (4, 100, 0.098), (5, 100, 0.112)],
)
def test_the_wilson_bounds_behind_the_pooling_rule(k: int, n: int, high: float) -> None:
    assert wilson_interval(k, n)[1] == pytest.approx(high, abs=0.0005)


def test_the_wilson_interval_of_the_judge_trial() -> None:
    low, high = wilson_interval(2, 30)  # 2 relevant out of 30 pairs, 2026-11-02

    assert (low, high) == (pytest.approx(0.0185, abs=0.0001), pytest.approx(0.2133, abs=0.0001))


def test_no_error_still_leaves_room_for_some() -> None:
    low, high = wilson_interval(0, 100)

    assert low == 0
    assert high > 0


def test_the_wilson_interval_is_symmetric() -> None:
    low, high = wilson_interval(3, 40)

    assert wilson_interval(37, 40) == (pytest.approx(1 - high), pytest.approx(1 - low))


def test_no_observation_tells_nothing() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize(("k", "n"), [(-1, 10), (11, 10)])
def test_an_impossible_count_is_refused(k: int, n: int) -> None:
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(k, n)
