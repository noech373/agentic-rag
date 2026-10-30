from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.pipeline_config import PipelineConfig, load_pipeline

PIPELINES = Path(__file__).parents[1] / "pipelines"


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


@pytest.mark.parametrize("path", sorted(PIPELINES.glob("*.toml")), ids=lambda path: path.stem)
def test_every_pipeline_file_of_the_repo_loads(path: Path) -> None:
    load_pipeline(path)


def test_the_repo_has_a_pipeline_per_retriever_besides_the_baseline() -> None:
    retrievers = {load_pipeline(path).retriever for path in PIPELINES.glob("*.toml")}

    assert retrievers == {"bm25", "hybrid"}
