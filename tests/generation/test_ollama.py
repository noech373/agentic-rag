import json
from typing import Any

import httpx
import pytest

from agentic_rag.document import Chunk, IndexedDocument, RetrievedChunk
from agentic_rag.generation.base import GenerationError, Generator
from agentic_rag.generation.ollama import OllamaGenerator
from agentic_rag.generation.prompt import INSTRUCTIONS, build_input

_NUM_CTX = 8000


class _FakeOllama:
    """The native chat endpoint, offline: answers with a given text, end and prompt size."""

    def __init__(
        self,
        text: str = "Within 72 hours [1].",
        done_reason: str = "stop",
        prompt_tokens: int = 900,
    ) -> None:
        self.text = text
        self.done_reason = done_reason
        self.prompt_tokens = prompt_tokens
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
                "prompt_eval_count": self.prompt_tokens,
                "eval_count": 30,
            },
        )


def _sources() -> tuple[RetrievedChunk, ...]:
    document = IndexedDocument(
        id="gdpr",
        title="GDPR",
        short_title="GDPR",
        source_url=None,
        regulations=("GDPR",),
        groups=("public",),
        sha256="0" * 64,
        embedding_model="fake@2",
    )
    chunk = Chunk(kind="text", text="Not later than 72 hours.", path=())
    return (RetrievedChunk(document=document, position=0, chunk=chunk, score=1.0),)


def _generate(fake: _FakeOllama) -> str:
    generator = OllamaGenerator(fake.client(), "qwen-test", _NUM_CTX)
    return generator.generate("Breach deadline?", _sources()).text


def test_it_is_a_generator() -> None:
    generator: Generator = OllamaGenerator(_FakeOllama().client(), "qwen-test", _NUM_CTX)

    answer = generator.generate("Breach deadline?", _sources())

    assert answer.text == "Within 72 hours [1]."
    assert answer.sources == _sources()


def test_the_request_holds_the_same_prompt_as_every_llm() -> None:
    fake = _FakeOllama()

    _generate(fake)

    assert fake.body()["model"] == "qwen-test"
    assert fake.body()["messages"] == [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": build_input("Breach deadline?", _sources())},
    ]


def test_the_request_asks_for_the_most_likely_words_in_the_loaded_window() -> None:
    fake = _FakeOllama()

    _generate(fake)

    assert fake.body()["options"] == {"num_ctx": _NUM_CTX, "temperature": 0}


def test_a_cut_answer_is_an_error_not_a_truncated_answer() -> None:
    with pytest.raises(GenerationError, match="length"):
        _generate(_FakeOllama(done_reason="length"))


def test_sources_that_filled_the_window_are_an_error_not_an_answer_from_half_of_them() -> None:
    with pytest.raises(GenerationError, match="truncated"):
        _generate(_FakeOllama(prompt_tokens=_NUM_CTX))
