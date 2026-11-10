import json
from typing import Any

import httpx
import pytest

from agentic_rag.generation.base import GenerationError
from agentic_rag.hyde.base import AnswerWriter
from agentic_rag.hyde.ollama import OllamaAnswerWriter
from agentic_rag.hyde.prompt import INSTRUCTIONS

QUESTION = "Our privacy person keeps blocking projects. Can we let her go for it?"
ANSWER = "The data protection officer shall not be dismissed or penalised for performing tasks."


class _FakeOllama:
    """The native chat endpoint, offline: answers with a given text and end."""

    def __init__(self, text: str = ANSWER, done_reason: str = "stop") -> None:
        self.text = text
        self.done_reason = done_reason
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.Client:
        transport = httpx.MockTransport(self._handle)
        return httpx.Client(base_url="https://ollama.test", transport=transport)

    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[0].content)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "qwen-test",
                "message": {"role": "assistant", "content": self.text},
                "done": True,
                "done_reason": self.done_reason,
                "prompt_eval_count": 80,
                "eval_count": 60,
            },
        )


def _write(fake: _FakeOllama) -> str:
    return OllamaAnswerWriter(fake.client(), "qwen-test", 1000).write(QUESTION)


def test_it_is_an_answer_writer() -> None:
    writer: AnswerWriter = OllamaAnswerWriter(_FakeOllama().client(), "qwen-test", 1000)

    assert writer.write(QUESTION) == ANSWER


def test_its_name_is_the_model() -> None:
    assert OllamaAnswerWriter(_FakeOllama().client(), "qwen-test", 1000).name == "qwen-test"


def test_the_model_reads_the_instructions_and_the_question_alone() -> None:
    """No document: the answer comes from the model's own knowledge."""
    fake = _FakeOllama()

    _write(fake)

    assert fake.body()["model"] == "qwen-test"
    assert fake.body()["messages"] == [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": QUESTION},
    ]


def test_the_request_asks_for_the_most_likely_words_in_the_loaded_window() -> None:
    fake = _FakeOllama()

    _write(fake)

    assert fake.body()["options"] == {"num_ctx": 1000, "temperature": 0}
    assert fake.body()["think"] is False


def test_an_answer_cut_by_its_length_is_an_error_not_half_an_answer() -> None:
    with pytest.raises(GenerationError, match="length"):
        _write(_FakeOllama(done_reason="length"))


def test_an_empty_answer_is_an_error_not_a_search_for_nothing() -> None:
    with pytest.raises(GenerationError, match="without text"):
        _write(_FakeOllama(text=" \n"))


def test_the_instructions_forbid_article_numbers() -> None:
    """The embedded titles hold article numbers: an invented one would pull the wrong article."""
    assert "Do not cite article" in INSTRUCTIONS
