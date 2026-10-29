import pytest

from agentic_rag.evaluation.comparison import (
    IncomparableRunsError,
    check_comparable,
    comparison_table,
)
from agentic_rag.evaluation.record import RunRecord
from agentic_rag.evaluation.runner import QuestionResult

_BASELINE = RunRecord(
    golden_sha256="9f2c",
    git_sha="e87c5ad0000",
    dirty=False,
    config={"retriever": "vector"},
    results=(),
)


def test_runs_that_differ_only_by_config_compare_without_warning() -> None:
    candidate = _BASELINE.model_copy(update={"config": {"retriever": "hybrid"}})

    assert check_comparable(_BASELINE, candidate) == []


def test_runs_on_different_golden_sets_are_refused() -> None:
    candidate = _BASELINE.model_copy(update={"golden_sha256": "1b7a"})

    with pytest.raises(IncomparableRunsError, match="golden"):
        check_comparable(_BASELINE, candidate)


def test_runs_from_different_commits_ask_to_rerun_the_baseline() -> None:
    candidate = _BASELINE.model_copy(update={"git_sha": "0842afe0000"})

    [warning] = check_comparable(_BASELINE, candidate)

    assert "e87c5ad" in warning
    assert "0842afe" in warning
    assert "rerun the baseline" in warning


@pytest.mark.parametrize(("dirty_baseline", "dirty_candidate"), [(True, False), (False, True)])
def test_a_dirty_run_is_not_for_the_readme(dirty_baseline: bool, dirty_candidate: bool) -> None:
    baseline = _BASELINE.model_copy(update={"dirty": dirty_baseline})
    candidate = _BASELINE.model_copy(update={"dirty": dirty_candidate})

    [warning] = check_comparable(baseline, candidate)

    assert "not for the README" in warning
    assert ("baseline" in warning) == dirty_baseline


def test_every_difference_gets_its_own_warning() -> None:
    candidate = _BASELINE.model_copy(update={"git_sha": "0842afe0000", "dirty": True})

    assert len(check_comparable(_BASELINE, candidate)) == 2


def _result(question_id: str, score: float) -> QuestionResult:
    recall = {f"recall@{k}": score for k in (5, 8, 20, 50)}
    return QuestionResult(
        question_id=question_id, category="factual", recall=recall, mrr=score, retrieved=()
    )


def _run(*scores: float) -> RunRecord:
    results = tuple(_result(f"q{i}", score) for i, score in enumerate(scores))
    return _BASELINE.model_copy(update={"results": results})


def _rows(table: str) -> dict[str, list[str]]:
    rows = [line.strip("|").split("|") for line in table.splitlines() if line.startswith("| ")]
    return {cells[0].strip(): [c.strip() for c in cells[1:]] for cells in rows}


def test_the_header_names_the_variants() -> None:
    table = comparison_table(_run(0, 1), _run(1, 1), "baseline", "bm25")

    assert table.splitlines()[0] == "| metric | baseline | bm25 | difference [95% CI] |"


def test_one_row_per_metric_in_order() -> None:
    table = comparison_table(_run(0, 1), _run(1, 1), "baseline", "bm25")

    metrics = [m for m in _rows(table) if m != "metric"]
    assert metrics == ["recall@5", "recall@8", "recall@20", "recall@50", "mrr"]


def test_a_row_holds_both_means_and_the_difference() -> None:
    table = comparison_table(_run(0, 0.5), _run(0.5, 1), "baseline", "bm25")

    assert _rows(table)["recall@8"] == ["0.25", "0.75", "**+0.50 [+0.50, +0.50]**"]


def test_a_difference_whose_interval_holds_zero_is_not_bold() -> None:
    table = comparison_table(_run(0, 1), _run(1, 0), "baseline", "bm25")

    [baseline, candidate, difference] = _rows(table)["recall@8"]
    assert (baseline, candidate) == ("0.50", "0.50")
    assert difference.startswith("+0.00 [-")
    assert "**" not in difference


def test_the_caption_gives_the_number_of_questions() -> None:
    table = comparison_table(_run(0, 1, 1), _run(1, 1, 1), "baseline", "bm25")

    assert table.splitlines()[-1].startswith("3 questions.")
