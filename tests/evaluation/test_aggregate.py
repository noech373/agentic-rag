from statistics import fmean

import pytest

from agentic_rag.evaluation.aggregate import (
    bootstrap_ci,
    mean_scores,
    mean_scores_by_category,
    paired_differences,
)
from agentic_rag.evaluation.golden import Category
from agentic_rag.evaluation.runner import QuestionResult


def _result(
    recall_at_8: float, mrr: float, id: str = "q", category: Category = "factual"
) -> QuestionResult:
    return QuestionResult(
        question_id=id,
        category=category,
        recall={"recall@8": recall_at_8},
        mrr=mrr,
        retrieved=(),
    )


def test_the_mean_is_taken_per_metric() -> None:
    results = [_result(1.0, 1.0, "a"), _result(0.5, 0.0, "b")]

    assert mean_scores(results) == {"recall@8": 0.75, "mrr": 0.5}


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
        "factual": {"recall@8": 0.75, "mrr": 0.75},
        "table": {"recall@8": 0.0, "mrr": 0.0},
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
    assert paired_differences(baseline, candidate, "mrr") == [0.5, 0.0]


def test_questions_are_paired_by_id_not_by_position() -> None:
    baseline = [_result(0.5, 0.0, "a"), _result(1.0, 1.0, "b")]
    candidate = [_result(0.5, 1.0, "b"), _result(1.0, 0.5, "a")]

    assert paired_differences(baseline, candidate, "recall@8") == [0.5, -0.5]


def test_runs_on_different_questions_cannot_be_paired() -> None:
    with pytest.raises(ValueError, match="same questions"):
        paired_differences([_result(1.0, 1.0, "a")], [_result(1.0, 1.0, "b")], "mrr")


def test_a_run_compared_with_itself_has_no_difference_at_all() -> None:
    run = [_result(0.5, 0.25, "a"), _result(1.0, 1.0, "b")]

    differences = paired_differences(run, run, "mrr")

    assert differences == [0.0, 0.0]
    assert bootstrap_ci(differences) == (0.0, 0.0)
