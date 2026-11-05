from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from agentic_rag.reranking import cross_encoder
from agentic_rag.reranking.base import Reranker
from agentic_rag.reranking.cross_encoder import CrossEncoderReranker


class _FakeModel:
    """Stands for the real model: scores from a table, and remembers how it was called."""

    created: list[tuple[str, dict[str, Any]]]
    scores: dict[str, float]
    predictions: list[tuple[list[tuple[str, str]], dict[str, Any]]]

    def __init__(self, name: str, **options: Any) -> None:
        type(self).created.append((name, options))

    def predict(self, pairs: Sequence[tuple[str, str]], **options: Any) -> np.ndarray:
        type(self).predictions.append((list(pairs), options))
        return np.array([self.scores.get(text, 0.0) for _, text in pairs], dtype=np.float32)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> type[_FakeModel]:
    _FakeModel.created = []
    _FakeModel.scores = {}
    _FakeModel.predictions = []
    monkeypatch.setattr(cross_encoder, "CrossEncoder", _FakeModel)
    return _FakeModel


def test_it_is_a_reranker(model: type[_FakeModel]) -> None:
    reranker: Reranker = CrossEncoderReranker("some/model")

    assert reranker.score("question", []) == []


def test_it_loads_the_model_it_is_given(model: type[_FakeModel]) -> None:
    CrossEncoderReranker("BAAI/bge-reranker-v2-m3")

    assert [name for name, _ in model.created] == ["BAAI/bge-reranker-v2-m3"]


def test_it_runs_on_the_cpu_and_reads_at_most_512_tokens(model: type[_FakeModel]) -> None:
    """The default of v2-m3 is 8192; the measures are at 512, on a machine without a GPU."""
    CrossEncoderReranker("some/model")

    _, options = model.created[0]
    assert options["device"] == "cpu"
    assert options["max_length"] == 512


def test_each_text_is_paired_with_the_question(model: type[_FakeModel]) -> None:
    CrossEncoderReranker("some/model").score("Can we dismiss our DPO?", ["text 1", "text 2"])

    pairs, _ = model.predictions[0]
    assert pairs == [("Can we dismiss our DPO?", "text 1"), ("Can we dismiss our DPO?", "text 2")]


def test_the_scores_stay_in_the_order_of_the_texts(model: type[_FakeModel]) -> None:
    model.scores = {"low": 0.25, "high": 0.75}

    scores = CrossEncoderReranker("some/model").score("question", ["low", "high"])

    assert scores == pytest.approx([0.25, 0.75])


def test_the_scores_are_python_floats(model: type[_FakeModel]) -> None:
    """The model gives a numpy array: the protocol, and the JSON of a run, want plain floats."""
    scores = CrossEncoderReranker("some/model").score("question", ["text"])

    assert isinstance(scores, list)
    assert type(scores[0]) is float


def test_it_scores_8_pairs_at_a_time(model: type[_FakeModel]) -> None:
    """The batch size of the memory peak we measured (2.8 GB with v2-m3)."""
    CrossEncoderReranker("some/model").score("question", ["text"])

    _, options = model.predictions[0]
    assert options["batch_size"] == 8


def test_no_progress_bar_is_shown(model: type[_FakeModel]) -> None:
    """One bar per question would flood the output of an evaluation run."""
    CrossEncoderReranker("some/model").score("question", ["text"])

    _, options = model.predictions[0]
    assert options["show_progress_bar"] is False
