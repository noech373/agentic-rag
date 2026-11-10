import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.document import Chunk, IndexedDocument, RetrievedChunk
from agentic_rag.evaluation.record import (
    RunRecord,
    file_sha256,
    git_dirty,
    git_sha,
    load_run,
    record_run,
    save_run,
)
from agentic_rag.evaluation.runner import QuestionResult
from agentic_rag.pipeline_config import PipelineConfig


def _record() -> RunRecord:
    return RunRecord(
        golden_sha256="9f2c",
        judgments_sha256="3d1e",
        git_sha="e87c5ad",
        dirty=True,
        config={"retriever": "vector", "k": 8, "rerank": False, "weight": 0.5},
        results=(
            QuestionResult(
                question_id="a",
                category="factual",
                recall={"recall@8": 0.5},
                mrr=0.25,
                judged=0.5,
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


def test_a_config_value_left_empty_survives_the_file(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    record = _record().model_copy(update={"config": {"retriever": "vector", "reranker": None}})

    save_run(record, path)

    assert load_run(path).config == {"retriever": "vector", "reranker": None}


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


def test_rerunning_a_committed_run_does_not_make_the_repo_dirty(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run = repo / "eval" / "results" / "baseline.json"
    run.parent.mkdir(parents=True)
    run.write_text("{}")
    _git(repo, "add", "eval")
    _git(repo, "commit", "--quiet", "-m", "baseline run")

    run.write_text('{"rerun": true}')

    assert not git_dirty(repo)
    assert not git_dirty(repo / "eval")  # the path is from the repo root, wherever git runs


def test_outside_a_repo_git_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        git_sha(tmp_path)


_GOLDEN = """
[[questions]]
id = "kept"
category = "factual"
origin = "blind"
question = "When to notify?"
answer = "Without undue delay."
[[questions.evidence]]
any_of = [{ document = "gdpr", quote = "notify without undue delay" }]

[[questions]]
id = "held-out"
category = "factual"
origin = "blind"
holdout = true
question = "When to notify?"
answer = "Without undue delay."
[[questions.evidence]]
any_of = [{ document = "gdpr", quote = "notify without undue delay" }]
"""


class _NothingFound:
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return []


_JUDGMENTS = 'kept = [{ document = "gdpr", quote = "the purpose of this regulation" }]\n'


def _repo_with_golden(tmp_path: Path, judgments: str = "") -> tuple[Path, Path, Path]:
    """A repo with a golden set and its judgments, both committed."""
    repo = _repo(tmp_path)
    golden = repo / "golden.toml"
    golden.write_text(_GOLDEN)
    judged = repo / "judgments.toml"
    judged.write_text(judgments)
    _git(repo, "add", "golden.toml", "judgments.toml")
    _git(repo, "commit", "--quiet", "-m", "golden set")
    return repo, golden, judged


def test_a_run_scores_only_the_tuning_questions(tmp_path: Path) -> None:
    repo, golden, judgments = _repo_with_golden(tmp_path)

    record = record_run(
        golden, judgments, lambda _: _NothingFound(), PipelineConfig(), repo, corpus_sha256="abc"
    )

    assert [r.question_id for r in record.results] == ["kept"]


def test_a_run_keeps_its_provenance(tmp_path: Path) -> None:
    repo, golden, judgments = _repo_with_golden(tmp_path)

    record = record_run(
        golden, judgments, lambda _: _NothingFound(), PipelineConfig(), repo, corpus_sha256="abc"
    )

    assert record.golden_sha256 == file_sha256(golden)
    assert record.judgments_sha256 == file_sha256(judgments)
    assert record.git_sha == _git(repo, "rev-parse", "HEAD")
    assert not record.dirty


def test_a_run_keeps_the_fingerprint_of_its_corpus(tmp_path: Path) -> None:
    repo, golden, judgments = _repo_with_golden(tmp_path)

    record = record_run(
        golden, judgments, lambda _: _NothingFound(), PipelineConfig(), repo, corpus_sha256="abc"
    )

    assert record.corpus_sha256 == "abc"


def test_a_run_saved_before_corpus_variants_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    saved = json.loads(_record().model_dump_json())
    del saved["corpus_sha256"]
    path.write_text(json.dumps(saved))

    assert load_run(path).corpus_sha256 is None


def test_a_run_keeps_the_validated_config_with_its_defaults(tmp_path: Path) -> None:
    repo, golden, judgments = _repo_with_golden(tmp_path)

    record = record_run(
        golden, judgments, lambda _: _NothingFound(), PipelineConfig(), repo, corpus_sha256="abc"
    )

    assert record.config == {
        "corpus": "titles",
        "embedded_text": "titles",
        "contextualizer": "openai",
        "retriever": "vector",
        "hyde": None,
        "reranker": None,  # None: no reranker, said
        "generator": "openai",
    }


def test_the_retriever_is_built_from_the_recorded_config(tmp_path: Path) -> None:
    repo, golden, judgments = _repo_with_golden(tmp_path)
    built_from: list[PipelineConfig] = []

    def make(pipeline: PipelineConfig) -> _NothingFound:
        built_from.append(pipeline)
        return _NothingFound()

    record = record_run(golden, judgments, make, PipelineConfig(), repo, corpus_sha256="abc")

    assert [p.model_dump() for p in built_from] == [record.config]


class _RejectedFound:
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        document = IndexedDocument(
            id="gdpr",
            title="GDPR",
            short_title="GDPR",
            source_url=None,
            regulations=(),
            groups=("public",),
            sha256="0",
            embedding_model="m",
        )
        chunk = Chunk(kind="text", text="The purpose of this Regulation is to protect.", path=())
        return [RetrievedChunk(document=document, position=0, chunk=chunk, score=1.0)]


@pytest.mark.parametrize(("judgments", "judged"), [("", 0), (_JUDGMENTS, 1)])
def test_a_run_scores_judged_on_the_judgments_file(
    tmp_path: Path, judgments: str, judged: float
) -> None:
    repo, golden, path = _repo_with_golden(tmp_path, judgments)

    record = record_run(
        golden, path, lambda _: _RejectedFound(), PipelineConfig(), repo, corpus_sha256="abc"
    )

    assert [r.judged for r in record.results] == [judged]
