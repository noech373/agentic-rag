import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.evaluation.record import (
    RunRecord,
    file_sha256,
    git_dirty,
    git_sha,
    load_run,
    save_run,
)
from agentic_rag.evaluation.runner import QuestionResult


def _record() -> RunRecord:
    return RunRecord(
        golden_sha256="9f2c",
        git_sha="e87c5ad",
        dirty=True,
        config={"retriever": "vector", "k": 8, "rerank": False, "weight": 0.5},
        results=(
            QuestionResult(
                question_id="a",
                category="factual",
                recall={"recall@8": 0.5},
                mrr=0.25,
                retrieved=("gdpr#12", "gdpr#13"),
            ),
        ),
    )


def test_a_saved_run_loads_back_identical(tmp_path: Path) -> None:
    path = tmp_path / "run.json"

    save_run(_record(), path)

    assert load_run(path) == _record()


def test_saving_creates_the_missing_directories(tmp_path: Path) -> None:
    path = tmp_path / "eval" / "results" / "run.json"

    save_run(_record(), path)

    assert path.is_file()


def test_the_file_is_json_that_can_be_read_by_hand(tmp_path: Path) -> None:
    path = tmp_path / "run.json"

    save_run(_record(), path)

    assert json.loads(path.read_text())["git_sha"] == "e87c5ad"
    assert "\n" in path.read_text().strip()  # indented, not one long line


def test_config_values_keep_their_type(tmp_path: Path) -> None:
    path = tmp_path / "run.json"

    save_run(_record(), path)

    # True == 1 in Python: compare types, not values
    assert {k: type(v) for k, v in load_run(path).config.items()} == {
        "retriever": str,
        "k": int,
        "rerank": bool,
        "weight": float,
    }


def test_an_unknown_field_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    data = json.loads(_record().model_dump_json())
    data["golden_sha"] = "typo"
    path.write_text(json.dumps(data))

    with pytest.raises(ValidationError):
        load_run(path)


def test_a_file_that_is_not_a_run_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text('{"results": []}')

    with pytest.raises(ValidationError):
        load_run(path)


def test_the_hash_is_the_sha256_of_the_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "golden.toml"
    path.write_bytes(b"[[questions]]\n")

    assert file_sha256(path) == hashlib.sha256(b"[[questions]]\n").hexdigest()


def test_any_edit_changes_the_hash(tmp_path: Path) -> None:
    path = tmp_path / "golden.toml"
    path.write_text("[[questions]]\n")
    before = file_sha256(path)

    path.write_text("[[questions]]\n# a comment\n")

    assert file_sha256(path) != before


def _git(repo: Path, *args: str) -> str:
    identity = ("-c", "user.name=Test", "-c", "user.email=test@example.com")
    result = subprocess.run(  # noqa: S603
        ["git", *identity, *args],  # noqa: S607
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "--quiet")
    (tmp_path / "code.py").write_text("x = 1\n")
    _git(tmp_path, "add", "code.py")
    _git(tmp_path, "commit", "--quiet", "-m", "first")
    return tmp_path


def test_the_sha_is_the_last_commit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    assert git_sha(repo) == _git(repo, "rev-parse", "HEAD")


def test_a_clean_repo_is_not_dirty(tmp_path: Path) -> None:
    assert not git_dirty(_repo(tmp_path))


def test_a_modified_tracked_file_makes_the_repo_dirty(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    (repo / "code.py").write_text("x = 2\n")

    assert git_dirty(repo)


def test_an_untracked_file_does_not_make_the_repo_dirty(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    (repo / "run.json").write_text("{}")

    assert not git_dirty(repo)


def test_outside_a_repo_git_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        git_sha(tmp_path)
