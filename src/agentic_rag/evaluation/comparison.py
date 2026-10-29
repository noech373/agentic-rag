"""Compare a run with the baseline: only the pipeline config should differ between them."""

from agentic_rag.evaluation.aggregate import bootstrap_ci, mean_scores, paired_differences
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


def summary_table(run: RunRecord, name: str) -> str:
    """One Markdown row per metric: the mean with its 95% CI, for a run seen alone."""
    means = mean_scores(run.results)
    lines = [f"| metric | {name} [95% CI] |", "|---|---|"]
    for metric, mean in means.items():
        low, high = bootstrap_ci([{**r.recall, "mrr": r.mrr}[metric] for r in run.results])
        lines.append(f"| {metric} | {mean:.2f} [{low:.2f}, {high:.2f}] |")
    lines += ["", f"{len(run.results)} questions."]
    return "\n".join(lines)


def comparison_table(
    baseline: RunRecord, candidate: RunRecord, baseline_name: str, candidate_name: str
) -> str:
    """One Markdown row per metric: both means and the paired difference with its 95% CI."""
    before = mean_scores(baseline.results)
    after = mean_scores(candidate.results)
    lines = [
        f"| metric | {baseline_name} | {candidate_name} | difference [95% CI] |",
        "|---|---|---|---|",
    ]
    for metric in before:
        differences = paired_differences(baseline.results, candidate.results, metric)
        low, high = bootstrap_ci(differences)
        difference = f"{after[metric] - before[metric]:+.2f} [{low:+.2f}, {high:+.2f}]"
        if low > 0 or high < 0:
            difference = f"**{difference}**"
        lines.append(f"| {metric} | {before[metric]:.2f} | {after[metric]:.2f} | {difference} |")
    lines += [
        "",
        f"{len(baseline.results)} questions. Bold: the 95% interval of the difference excludes 0.",
    ]
    return "\n".join(lines)
