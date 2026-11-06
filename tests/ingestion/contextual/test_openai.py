import json
from typing import Any

import httpx
import pytest
from openai import OpenAI

from agentic_rag.document import Chunk, Division
from agentic_rag.generation.base import GenerationError
from agentic_rag.ingestion.contextual.openai import OpenAIContextualizer
from agentic_rag.ingestion.contextual.prompt import INSTRUCTIONS, build_input

_DOCUMENT = "<document>...</document>"
_CHUNK = Chunk(
    kind="text",
    text="Not later than 72 hours.",
    path=(Division(kind="article", number="33", title="Notification"),),
)


class _FakeOpenAI:
    """The Responses endpoint, offline: answers with a given text and status."""

    def __init__(self, text: str = "Part of Article 33.", status: str = "completed") -> None:
        self.text = text
        self.status = status
        self.requests: list[dict[str, Any]] = []

    def client(self) -> OpenAI:
        transport = httpx.MockTransport(self._handle)
        # The SDK runs on httpx2 and also accepts an httpx client; its annotation names only httpx2.
        http_client = httpx.Client(transport=transport)
        return OpenAI(api_key="sk-test", http_client=http_client, max_retries=0)  # pyright: ignore[reportArgumentType]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        message: dict[str, Any] = {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": self.status,
            "content": [{"type": "output_text", "text": self.text, "annotations": []}],
        }
        incomplete = {"reason": "max_output_tokens"} if self.status == "incomplete" else None
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": body["model"],
                "status": self.status,
                "incomplete_details": incomplete,
                "output": [message],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        )


def _situate(fake: _FakeOpenAI) -> str:
    return OpenAIContextualizer(fake.client(), "gpt-test").situate(_DOCUMENT, _CHUNK)


def test_its_name_is_the_model() -> None:
    assert OpenAIContextualizer(_FakeOpenAI().client(), "gpt-test").name == "gpt-test"


def test_the_request_holds_the_model_the_instructions_and_the_input() -> None:
    fake = _FakeOpenAI()

    _situate(fake)

    assert fake.requests[0]["model"] == "gpt-test"
    assert fake.requests[0]["instructions"] == INSTRUCTIONS
    assert fake.requests[0]["input"] == build_input(_DOCUMENT, _CHUNK)


def test_openai_keeps_no_copy_of_the_response() -> None:
    fake = _FakeOpenAI()

    _situate(fake)

    assert fake.requests[0]["store"] is False


def test_the_context_is_the_text_of_the_response() -> None:
    assert _situate(_FakeOpenAI(text="Part of Article 33.")) == "Part of Article 33."


def test_an_incomplete_response_is_an_error_not_a_truncated_context() -> None:
    with pytest.raises(GenerationError, match="max_output_tokens"):
        _situate(_FakeOpenAI(status="incomplete"))


@pytest.mark.parametrize("text", ["", "  \n"])
def test_a_response_without_text_is_an_error_not_an_empty_context(text: str) -> None:
    with pytest.raises(GenerationError, match="without text"):
        _situate(_FakeOpenAI(text=text))
