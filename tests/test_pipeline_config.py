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


def test_the_default_config_has_no_reranker() -> None:
    assert PipelineConfig().reranker is None


@pytest.mark.parametrize("name", ["bge-reranker-v2-m3", "ms-marco-minilm"])
def test_a_reranker_is_read_from_a_file(tmp_path: Path, name: str) -> None:
    path = _write(tmp_path, f'reranker = "{name}"\n')

    assert load_pipeline(path).reranker == name


def test_an_unknown_reranker_is_refused_when_the_file_is_read(tmp_path: Path) -> None:
    # at loading, not after the model has been downloaded
    path = _write(tmp_path, 'reranker = "bge-reranker-v3"\n')

    with pytest.raises(ValidationError):
        load_pipeline(path)


def test_the_default_config_searches_the_titles_corpus() -> None:
    assert PipelineConfig().corpus == "titles"


@pytest.mark.parametrize("name", ["titles", "titles_v2", "c"])
def test_a_corpus_is_read_from_a_file(tmp_path: Path, name: str) -> None:
    path = _write(tmp_path, f'corpus = "{name}"\n')

    assert load_pipeline(path).corpus == name


@pytest.mark.parametrize(
    "name",
    ["", "Titles", "2titles", "ti-tles", "a.b", "titles\n", "x; DROP SCHEMA public CASCADE"],
)
def test_a_corpus_name_unsafe_in_sql_is_refused(name: str) -> None:
    # the name is written as is in CREATE SCHEMA and in the BM25 index name
    with pytest.raises(ValidationError):
        PipelineConfig(corpus=name)


def test_the_default_config_embeds_the_chunk_with_its_titles() -> None:
    assert PipelineConfig().embedded_text == "titles"


def test_the_embedded_text_is_read_from_a_file(tmp_path: Path) -> None:
    path = _write(tmp_path, 'embedded_text = "titles"\n')

    assert load_pipeline(path).embedded_text == "titles"


def test_an_unknown_embedded_text_is_refused_when_the_file_is_read(tmp_path: Path) -> None:
    path = _write(tmp_path, 'embedded_text = "title"\n')

    with pytest.raises(ValidationError, match="embedded_text"):
        load_pipeline(path)


def test_openai_writes_the_contexts_by_default() -> None:
    assert PipelineConfig().contextualizer == "openai"


def test_who_writes_the_contexts_is_read_from_a_file(tmp_path: Path) -> None:
    path = _write(tmp_path, 'contextualizer = "ollama"\n')

    assert load_pipeline(path).contextualizer == "ollama"


def test_an_unknown_contextualizer_is_refused_when_the_file_is_read(tmp_path: Path) -> None:
    path = _write(tmp_path, 'contextualizer = "qwen"\n')

    with pytest.raises(ValidationError, match="contextualizer"):
        load_pipeline(path)


def test_the_default_config_searches_with_the_question() -> None:
    assert PipelineConfig().hyde is None


def test_who_writes_the_hyde_answers_is_read_from_a_file(tmp_path: Path) -> None:
    assert load_pipeline(_write(tmp_path, 'hyde = "ollama"')).hyde == "ollama"


def test_hyde_without_a_writer_is_refused(tmp_path: Path) -> None:
    """`true` would not say which LLM writes the answers."""
    with pytest.raises(ValidationError, match="hyde"):
        load_pipeline(_write(tmp_path, "hyde = true"))


def test_openai_writes_the_answers_by_default() -> None:
    assert PipelineConfig().generator == "openai"


def test_who_writes_the_answers_is_read_from_a_file(tmp_path: Path) -> None:
    assert load_pipeline(_write(tmp_path, 'generator = "ollama"')).generator == "ollama"


def test_an_unknown_generator_is_refused_when_the_file_is_read(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="generator"):
        load_pipeline(_write(tmp_path, 'generator = "mistral"'))


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


def test_the_repo_has_a_pipeline_per_variant() -> None:
    variants = {
        (config.corpus, config.embedded_text, config.retriever, config.hyde, config.reranker)
        for config in map(load_pipeline, PIPELINES.glob("*.toml"))
    }

    assert variants == {
        ("public", "chunk", "vector", None, None),
        ("titles", "titles", "bm25", None, None),
        ("titles", "titles", "hybrid", None, None),
        ("titles", "titles", "vector", None, "bge-reranker-v2-m3"),
        ("titles", "titles", "vector", None, None),
        ("titles", "titles", "vector", "ollama", None),
        ("contextual", "context", "vector", None, None),
    }
