import base64
import json
from array import array
from typing import Any

import httpx
import pytest
from openai import OpenAI

from agentic_rag.embeddings import OpenAIEmbedder


class _FakeOpenAI:
    """The embeddings endpoint, offline: the vector of a text is [its length, its index]."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def client(self) -> OpenAI:
        transport = httpx.MockTransport(self._handle)
        # The SDK runs on httpx2 and also accepts an httpx client; its annotation names only httpx2.
        http_client = httpx.Client(transport=transport)
        return OpenAI(api_key="sk-test", http_client=http_client)  # pyright: ignore[reportArgumentType]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        data = [
            {"object": "embedding", "index": i, "embedding": _base64([len(text), i])}
            for i, text in enumerate(body["input"])
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": data[::-1],  # the API gives an index with each vector: don't rely on order
                "model": body["model"],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            },
        )


def _base64(vector: list[int]) -> str:
    """What the SDK asks for by default: little-endian float32, base64-encoded."""
    return base64.b64encode(array("f", vector).tobytes()).decode()


@pytest.fixture
def api() -> _FakeOpenAI:
    return _FakeOpenAI()


def _embedder(api: _FakeOpenAI) -> OpenAIEmbedder:
    return OpenAIEmbedder(api.client(), "text-embedding-3-small", 1536)


def test_the_name_identifies_the_model_and_its_dimensions(api: _FakeOpenAI) -> None:
    assert _embedder(api).name == "text-embedding-3-small@1536"


def test_embed_documents_returns_one_vector_per_text_in_order(api: _FakeOpenAI) -> None:
    vectors = _embedder(api).embed_documents(["a", "bbb", "cc"])

    assert vectors == [[1.0, 0.0], [3.0, 1.0], [2.0, 2.0]]
    assert len(api.requests) == 1
    assert api.requests[0]["model"] == "text-embedding-3-small"
    assert api.requests[0]["dimensions"] == 1536
    assert api.requests[0]["input"] == ["a", "bbb", "cc"]


def test_embed_query_returns_the_vector_of_the_text(api: _FakeOpenAI) -> None:
    assert _embedder(api).embed_query("abcd") == [4.0, 0.0]


def test_no_text_makes_no_request(api: _FakeOpenAI) -> None:
    assert _embedder(api).embed_documents([]) == []
    assert api.requests == []


def test_requests_hold_at_most_2048_texts(api: _FakeOpenAI) -> None:
    vectors = _embedder(api).embed_documents(["x"] * 2049)

    assert [len(request["input"]) for request in api.requests] == [2048, 1]
    assert len(vectors) == 2049


def test_requests_hold_at_most_300k_bytes_of_text(api: _FakeOpenAI) -> None:
    """A token is at least one byte: 300k bytes is at most the API's 300k tokens per request."""
    vectors = _embedder(api).embed_documents(["x" * 100_000] * 4)

    assert [len(request["input"]) for request in api.requests] == [3, 1]
    assert [vector[0] for vector in vectors] == [100_000.0] * 4


def test_bytes_are_counted_in_utf8(api: _FakeOpenAI) -> None:
    _embedder(api).embed_documents(["é" * 100_000] * 2)  # 2 bytes each in UTF-8

    assert [len(request["input"]) for request in api.requests] == [1, 1]


def test_a_text_over_the_limit_is_sent_alone(api: _FakeOpenAI) -> None:
    """Never an empty request: the API, not the batching, rejects a text that is too long."""
    _embedder(api).embed_documents(["short", "x" * 400_000, "short"])

    assert [len(request["input"]) for request in api.requests] == [1, 1, 1]
