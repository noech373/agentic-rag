from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_rag.cli import _load_pipeline, app  # pyright: ignore[reportPrivateUsage]
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
