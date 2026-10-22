import json
from typing import Any

import httpx
import pytest
from openai import OpenAI

from agentic_rag.document import Chunk
from agentic_rag.generation.base import GenerationError
from agentic_rag.generation.openai import OpenAIGenerator
from agentic_rag.generation.prompt import INSTRUCTIONS, build_input
from agentic_rag.storage.base import IndexedDocument, RetrievedChunk


class _FakeOpenAI:
    """The Responses endpoint, offline: answers with a given text and status."""

    def __init__(self, text: str = "Within 72 hours [1].", status: str = "completed") -> None:
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


def test_the_request_holds_the_instructions_the_sources_and_the_question() -> None:
    api = _FakeOpenAI()

    OpenAIGenerator(api.client(), "some-model").generate("Breach deadline?", _sources())

    (request,) = api.requests
    assert request["model"] == "some-model"
    assert request["instructions"] == INSTRUCTIONS
    assert request["input"] == build_input("Breach deadline?", _sources())


def test_openai_keeps_no_copy_of_the_response() -> None:
    api = _FakeOpenAI()

    OpenAIGenerator(api.client(), "some-model").generate("Breach deadline?", _sources())

    assert api.requests[0]["store"] is False


def test_the_answer_is_the_text_of_the_response_with_its_sources() -> None:
    api = _FakeOpenAI(text="Within 72 hours [1].")

    answer = OpenAIGenerator(api.client(), "some-model").generate("Breach deadline?", _sources())

    assert answer.text == "Within 72 hours [1]."
    assert answer.sources == _sources()


def test_an_incomplete_response_is_an_error_not_a_truncated_answer() -> None:
    api = _FakeOpenAI(text="Within 72", status="incomplete")

    with pytest.raises(GenerationError, match="max_output_tokens"):
        OpenAIGenerator(api.client(), "some-model").generate("Breach deadline?", _sources())
