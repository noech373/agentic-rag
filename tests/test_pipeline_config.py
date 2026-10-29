from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.pipeline_config import PipelineConfig, load_pipeline


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "variant.toml"
    path.write_text(text)
    return path


def test_the_default_config_is_the_plain_vector_rag() -> None:
    assert PipelineConfig().retriever == "vector"


def test_a_file_is_read_into_the_config(tmp_path: Path) -> None:
    path = _write(tmp_path, 'retriever = "vector"\n')

    assert load_pipeline(path) == PipelineConfig(retriever="vector")


def test_an_empty_file_is_the_default_config(tmp_path: Path) -> None:
    assert load_pipeline(_write(tmp_path, "")) == PipelineConfig()


def test_a_misspelled_key_is_refused(tmp_path: Path) -> None:
    # ignored, it would run the baseline under another name
    path = _write(tmp_path, "rerankr = true\n")

    with pytest.raises(ValidationError):
        load_pipeline(path)


def test_an_unknown_retriever_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, 'retriever = "vectr"\n')

    with pytest.raises(ValidationError):
        load_pipeline(path)


def test_the_config_cannot_be_changed_after_loading() -> None:
    config = PipelineConfig()

    with pytest.raises(ValidationError):
        config.retriever = "vector"  # pyright: ignore[reportAttributeAccessIssue]
