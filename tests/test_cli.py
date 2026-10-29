from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_rag.cli import _load_pipeline, app  # pyright: ignore[reportPrivateUsage]
from agentic_rag.evaluation import record
from agentic_rag.evaluation.record import RunRecord, load_run, save_run
from agentic_rag.evaluation.runner import QuestionResult
from agentic_rag.pipeline_config import PipelineConfig

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No .env and no keys: a command that reads the settings fails on them."""
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "POSTGRES_USER", "POSTGRES_PASSWORD"):
        monkeypatch.delenv(name, raising=False)


def test_no_pipeline_file_means_the_baseline() -> None:
    assert _load_pipeline(None) == PipelineConfig()


def test_a_pipeline_file_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "variant.toml"
    path.write_text('retriever = "vector"\n')

    assert _load_pipeline(path) == PipelineConfig(retriever="vector")


def test_a_bad_pipeline_file_is_a_usage_error_before_the_settings_are_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "variant.toml"
    path.write_text("rerankr = true\n")

    result = runner.invoke(app, ["ask", "What is a personal data breach?", "--pipeline", str(path)])

    assert result.exit_code == 2  # usage error, not a traceback
    assert "rerankr" in result.output
    assert "api_key" not in result.output


def test_a_missing_pipeline_file_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ask", "question", "--pipeline", str(tmp_path / "nope.toml")])

    assert result.exit_code == 2


def test_a_directory_is_not_a_pipeline_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ask", "question", "--pipeline", str(tmp_path)])

    assert result.exit_code == 2


def _run(score: float, **provenance: object) -> RunRecord:
    recall = {f"recall@{k}": score for k in (5, 8, 20, 50)}
    results = tuple(
        QuestionResult(
            question_id=f"q{i}", category="factual", recall=recall, mrr=score, retrieved=()
        )
        for i in range(3)
    )
    run = RunRecord(
        golden_sha256="9f2c",
        git_sha="e87c5ad0000",
        dirty=False,
        config={"retriever": "vector"},
        results=results,
    )
    return run.model_copy(update=provenance)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[PipelineConfig]:
    """Replace the retrieval with a fixed run of score 1; return the configs it was asked to run."""
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "golden.toml").write_text("")
    monkeypatch.setattr("agentic_rag.config.get_settings", lambda: None)
    configs: list[PipelineConfig] = []

    def fake_record_run(
        golden: Path, make_retriever: object, pipeline: PipelineConfig, repo: Path
    ) -> RunRecord:
        configs.append(pipeline)
        return _run(1.0)

    monkeypatch.setattr(record, "record_run", fake_record_run)
    return configs


def _variant(tmp_path: Path, name: str = "variant") -> str:
    path = tmp_path / f"{name}.toml"
    path.write_text('retriever = "vector"\n')
    return str(path)


def test_eval_without_pipeline_runs_and_saves_the_baseline(recorded: list[PipelineConfig]) -> None:
    result = runner.invoke(app, ["eval"])

    assert result.exit_code == 0, result.output
    assert recorded == [PipelineConfig()]
    assert load_run(record.RESULTS_DIR / "baseline.json") == _run(1.0)


def test_eval_shows_the_means_of_the_run(recorded: list[PipelineConfig]) -> None:
    result = runner.invoke(app, ["eval"])

    assert "| metric | baseline [95% CI] |" in result.stdout
    assert "| recall@8 | 1.00 [1.00, 1.00] |" in result.stdout


def test_eval_of_a_variant_is_saved_under_its_file_name(
    recorded: list[PipelineConfig], tmp_path: Path
) -> None:
    result = runner.invoke(app, ["eval", "--pipeline", _variant(tmp_path, "bm25")])

    assert result.exit_code == 0, result.output
    assert recorded == [PipelineConfig(retriever="vector")]
    assert (record.RESULTS_DIR / "bm25.json").is_file()
    assert "| metric | bm25 [95% CI] |" in result.stdout


def test_eval_does_not_compare_with_a_saved_baseline(
    recorded: list[PipelineConfig], tmp_path: Path
) -> None:
    save_run(_run(0.0), record.RESULTS_DIR / "baseline.json")

    result = runner.invoke(app, ["eval", "--pipeline", _variant(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "difference" not in result.stdout
    assert load_run(record.RESULTS_DIR / "baseline.json") == _run(0.0)  # left as it was


def test_rerunning_a_variant_replaces_its_run(recorded: list[PipelineConfig]) -> None:
    save_run(_run(0.0), record.RESULTS_DIR / "baseline.json")

    result = runner.invoke(app, ["eval"])

    assert result.exit_code == 0, result.output
    assert load_run(record.RESULTS_DIR / "baseline.json") == _run(1.0)


def test_a_bad_pipeline_file_stops_eval_before_any_run(
    recorded: list[PipelineConfig], tmp_path: Path
) -> None:
    path = tmp_path / "variant.toml"
    path.write_text("rerankr = true\n")

    result = runner.invoke(app, ["eval", "--pipeline", str(path)])

    assert result.exit_code == 2
    assert recorded == []


def test_eval_needs_the_golden_set(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "--golden", str(tmp_path / "nope.toml")])

    assert result.exit_code == 2


def _saved(tmp_path: Path, name: str, run: RunRecord) -> str:
    path = tmp_path / f"{name}.json"
    save_run(run, path)
    return str(path)


def test_compare_shows_the_difference_between_two_saved_runs(tmp_path: Path) -> None:
    before = _saved(tmp_path, "bm25", _run(0.0))
    after = _saved(tmp_path, "bm25-rerank", _run(1.0))

    result = runner.invoke(app, ["compare", before, after])

    assert result.exit_code == 0, result.output
    assert "| metric | bm25 | bm25-rerank | difference [95% CI] |" in result.stdout
    assert "| recall@8 | 0.00 | 1.00 | **+1.00 [+1.00, +1.00]** |" in result.stdout
    assert result.stderr == ""


def test_compare_warns_on_stderr(tmp_path: Path) -> None:
    before = _saved(tmp_path, "baseline", _run(0.0, git_sha="0842afe0000", dirty=True))
    after = _saved(tmp_path, "bm25", _run(1.0))

    result = runner.invoke(app, ["compare", before, after])

    assert result.exit_code == 0, result.output
    assert "difference [95% CI]" in result.stdout
    assert "rerun the baseline" in result.stderr
    assert "not for the README" in result.stderr


def test_compare_refuses_runs_on_different_golden_sets(tmp_path: Path) -> None:
    before = _saved(tmp_path, "baseline", _run(0.0, golden_sha256="1b7a"))
    after = _saved(tmp_path, "bm25", _run(1.0))

    result = runner.invoke(app, ["compare", before, after])

    assert result.exit_code == 1
    assert "golden" in result.stderr
    assert result.stdout == ""


def test_compare_refuses_a_file_that_is_not_a_run(tmp_path: Path) -> None:
    before = _saved(tmp_path, "baseline", _run(0.0))
    other = tmp_path / "other.json"
    other.write_text("{}")

    result = runner.invoke(app, ["compare", before, str(other)])

    assert result.exit_code == 2
    assert "not a saved run" in result.output


def test_compare_needs_existing_files(tmp_path: Path) -> None:
    result = runner.invoke(app, ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")])

    assert result.exit_code == 2
