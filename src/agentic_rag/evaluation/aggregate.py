"""Summarize a run: from one result per question to one figure per metric."""

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from statistics import NormalDist, fmean

from agentic_rag.evaluation.golden import Category
from agentic_rag.evaluation.runner import QuestionResult


def mean_scores(results: Sequence[QuestionResult]) -> dict[str, float]:
    """The mean of each metric over the questions, each question weighing the same."""
    scores = [r.scores() for r in results]
    if not scores:
        raise ValueError("no results to average")
    return {metric: fmean(s[metric] for s in scores) for metric in scores[0]}


def mean_scores_by_category(results: Sequence[QuestionResult]) -> dict[Category, dict[str, float]]:
    """The mean of each metric, calculated separately for each present category."""
    results_by_category: dict[Category, list[QuestionResult]] = defaultdict(list)

    for result in results:
        results_by_category[result.category].append(result)

    return {category: mean_scores(results) for category, results in results_by_category.items()}


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """The interval holding `confidence` of the means of `values` redrawn with replacement."""
    if not values:
        raise ValueError("no scores to resample")

    rng = random.Random(seed)  # noqa: S311  (reproducible resampling, not security)
    means = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(n_resamples))

    tail = (1 - confidence) / 2  # the share cut at each end
    low = int(tail * n_resamples)
    high = min(round((1 - tail) * n_resamples), n_resamples - 1)
    return means[low], means[high]


def wilson_interval(k: int, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """The confidence interval of a proportion `k / n`, sound even when `k` is 0 or `n`.

    Unlike p ± z·√(p(1-p)/n), it never collapses to a single point: 0 errors out of 100 still
    leaves room for a few percent. With no observation, it is the whole of [0, 1].
    """
    if not 0 <= k <= n:
        raise ValueError(f"{k} successes out of {n} trials")
    if n == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _scores_by_question(results: Sequence[QuestionResult], metric: str) -> dict[str, float]:
    return {r.question_id: r.scores()[metric] for r in results}


def paired_differences(
    baseline: Sequence[QuestionResult], candidate: Sequence[QuestionResult], metric: str
) -> list[float]:
    """For each question, the candidate's score minus the baseline's, in the baseline's order."""
    before = _scores_by_question(baseline, metric)
    after = _scores_by_question(candidate, metric)
    if before.keys() != after.keys():
        raise ValueError("both runs must cover the same questions")
    return [after[question_id] - before[question_id] for question_id in before]
