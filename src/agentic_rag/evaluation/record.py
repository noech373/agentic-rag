"""Save a run to JSON and load it back, so two runs can be compared without running them again."""

import hashlib
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentic_rag.evaluation.runner import QuestionResult


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    golden_sha256: str
    git_sha: str
    dirty: bool  # uncommitted changes: the code may not match git_sha
    config: dict[str, str | int | float | bool]
    results: tuple[QuestionResult, ...]


def file_sha256(path: Path) -> str:
    """Hash of the raw bytes: any edit, even a comment, gives a new hash."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def git_dirty(repo: Path) -> bool:
    """Tracked files changed since the last commit; untracked ones (new runs) don't count."""
    return _git(repo, "status", "--porcelain", "--untracked-files=no") != ""


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
