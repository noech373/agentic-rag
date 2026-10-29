import pytest

from agentic_rag.evaluation.comparison import IncomparableRunsError, check_comparable
from agentic_rag.evaluation.record import RunRecord

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
