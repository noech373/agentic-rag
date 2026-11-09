import json
from typing import Any

import httpx
import pytest

from agentic_rag.document import Chunk, Division
from agentic_rag.generation.base import GenerationError
from agentic_rag.ingestion.contextual.ollama import OllamaContextualizer
from agentic_rag.ingestion.contextual.prompt import INSTRUCTIONS, build_input

_DOCUMENT = "<document>...</document>"
_CHUNK = Chunk(
    kind="text",
    text="Not later than 72 hours.",
    path=(Division(kind="article", number="33", title="Notification"),),
)
_NUM_CTX = 1000


class _FakeOllama:
    """The native chat endpoint, offline: answers with a given text, end and token counts."""

    def __init__(
        self,
        text: str = "Part of Article 33.",
        done_reason: str = "stop",
        prompt_tokens: int = 500,
        status: int = 200,
    ) -> None:
        self.text = text
        self.done_reason = done_reason
        self.prompt_tokens = prompt_tokens
        self.status = status
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.Client:
        transport = httpx.MockTransport(self._handle)
        return httpx.Client(base_url="https://ollama.test", transport=transport)

    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[0].content)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "model not found"})
        return httpx.Response(
            200,
            json={
                "model": "qwen-test",
                "message": {"role": "assistant", "content": self.text},
                "done": True,
                "done_reason": self.done_reason,
                "prompt_eval_count": self.prompt_tokens,
                "eval_count": 20,
            },
        )


def _situate(fake: _FakeOllama) -> str:
    return OllamaContextualizer(fake.client(), "qwen-test", _NUM_CTX).situate(_DOCUMENT, _CHUNK)


def test_its_name_is_the_model() -> None:
    assert OllamaContextualizer(_FakeOllama().client(), "qwen-test", _NUM_CTX).name == "qwen-test"


def test_it_asks_the_native_chat_endpoint_not_the_openai_compatible_one() -> None:
    fake = _FakeOllama()

    _situate(fake)

    assert str(fake.requests[0].url) == "https://ollama.test/api/chat"


def test_the_request_holds_the_model_the_instructions_and_the_input() -> None:
    fake = _FakeOllama()

    _situate(fake)

    assert fake.body()["model"] == "qwen-test"
    assert fake.body()["messages"] == [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": build_input(_DOCUMENT, _CHUNK)},
    ]


def test_the_request_sets_the_context_length_and_turns_thinking_and_streaming_off() -> None:
    fake = _FakeOllama()

    _situate(fake)

    assert fake.body()["options"] == {"num_ctx": _NUM_CTX}
    assert fake.body()["think"] is False
    assert fake.body()["stream"] is False


def test_the_context_is_the_text_of_the_response() -> None:
    assert _situate(_FakeOllama(text="Part of Article 33.")) == "Part of Article 33."


def test_a_response_cut_by_its_length_is_an_error_not_a_truncated_context() -> None:
    with pytest.raises(GenerationError, match="length"):
        _situate(_FakeOllama(done_reason="length"))


def test_a_prompt_that_filled_the_window_is_an_error_not_a_context_of_half_a_document() -> None:
    """Ollama cuts a prompt too long without an error: 990 tokens read + 20 written fill 1000."""
    with pytest.raises(GenerationError, match="truncated"):
        _situate(_FakeOllama(prompt_tokens=990))


def test_a_prompt_that_fits_the_window_is_accepted() -> None:
    assert _situate(_FakeOllama(prompt_tokens=979)) == "Part of Article 33."


@pytest.mark.parametrize("text", ["", "  \n"])
def test_a_response_without_text_is_an_error_not_an_empty_context(text: str) -> None:
    with pytest.raises(GenerationError, match="without text"):
        _situate(_FakeOllama(text=text))


def test_a_server_error_is_raised_not_turned_into_a_context() -> None:
    with pytest.raises(httpx.HTTPStatusError):
        _situate(_FakeOllama(status=404))
