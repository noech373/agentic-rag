"""Compare a run with the baseline: only the pipeline config should differ between them."""

from agentic_rag.evaluation.record import RunRecord


class IncomparableRunsError(ValueError):
    pass


def check_comparable(baseline: RunRecord, candidate: RunRecord) -> list[str]:
    """Raise if the questions differ; return a warning for each other difference."""
    if baseline.golden_sha256 != candidate.golden_sha256:
        raise IncomparableRunsError("the runs used different golden sets: rerun the baseline")

    warnings: list[str] = []
    if baseline.git_sha != candidate.git_sha:
        warnings.append(
            f"the runs come from different commits ({baseline.git_sha[:7]}, "
            f"{candidate.git_sha[:7]}): rerun the baseline so that only the config differs"
        )
    for name, record in (("baseline", baseline), ("candidate", candidate)):
        if record.dirty:
            warnings.append(f"the {name} ran on uncommitted changes: not for the README")
    return warnings
