"""A run of the golden set with its provenance, saved to JSON to compare runs without rerunning."""

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentic_rag.evaluation.golden import load_golden_set
from agentic_rag.evaluation.runner import QuestionResult, run_retriever, tuning_questions
from agentic_rag.pipeline_config import PipelineConfig
from agentic_rag.retrieval.base import Retriever

RESULTS_DIR = Path(
    "eval/results"
)  # from the repo root; committed, so the README tables can be rebuilt


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    golden_sha256: str
    git_sha: str
    dirty: bool  # uncommitted changes: the code may not match git_sha
    config: dict[str, str | int | float | bool]
    results: tuple[QuestionResult, ...]


def record_run(
    golden_path: Path,
    make_retriever: Callable[[PipelineConfig], Retriever],
    pipeline: PipelineConfig,
    repo: Path,
) -> RunRecord:
    """Run the retriever on the tuning questions and keep what makes the run comparable."""
    retriever = make_retriever(pipeline)  # from the config it records: they cannot differ
    results = run_retriever(tuning_questions(load_golden_set(golden_path)), retriever)
    return RunRecord(
        golden_sha256=file_sha256(golden_path),
        git_sha=git_sha(repo),
        dirty=git_dirty(repo),
        config=pipeline.model_dump(),  # validated, defaults included: what actually ran
        results=tuple(results),
    )


def file_sha256(path: Path) -> str:
    """Hash of the raw bytes: any edit, even a comment, gives a new hash."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def git_dirty(repo: Path) -> bool:
    """Tracked files changed since the last commit, saved runs aside: rerunning the baseline
    must not make the next run dirty. Untracked files (new runs) don't count either."""
    results = f":(top,exclude){RESULTS_DIR.as_posix()}"
    return _git(repo, "status", "--porcelain", "--untracked-files=no", "--", results) != ""


def _git(repo: Path, *args: str) -> str:
    # fixed arguments, no user input: the shell-injection warnings don't apply
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def save_run(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_run(path: Path) -> RunRecord:
    """Raise ValidationError if the file is not a run."""
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
