import json
from typing import Any

import httpx
import pytest
from openai import OpenAI

from agentic_rag.document import Chunk, Division, IndexedDocument, StoredChunk
from agentic_rag.evaluation.golden import Evidence, Question, Quote
from agentic_rag.evaluation.relevance_judge import (
    INSTRUCTIONS,
    JudgeError,
    OpenAIRelevanceJudge,
    StatedFact,
    Verdict,
    build_input,
    unsupported_facts,
)

_VERDICT = {
    "reason": "It gives the deadline.",
    "facts": [{"fact": 1, "quote": "within 72 hours"}],
}


class _FakeOpenAI:
    """The Responses endpoint, offline: answers with a given JSON text and status."""

    def __init__(self, verdict: dict[str, Any] = _VERDICT, status: str = "completed") -> None:
        self.text = json.dumps(verdict)
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


def _question() -> Question:
    return Question(
        id="breach-deadline",
        category="factual",
        origin="blind",
        question="How long do I have to report a breach?",
        answer="72 hours (GDPR, Article 33(1)).",
        evidence=(
            Evidence(
                any_of=(
                    Quote(document="gdpr", quote="not later than 72 hours after"),
                    Quote(document="edpb-guidelines-2022-09", quote="within 72 hours"),
                )
            ),
            Evidence(any_of=(Quote(document="gdpr", quote="without undue delay"),)),
        ),
    )


def _chunk() -> StoredChunk:
    document = IndexedDocument(
        id="gdpr",
        title="General Data Protection Regulation",
        short_title="GDPR",
        source_url=None,
        regulations=("GDPR",),
        groups=("public",),
        sha256="0",
        embedding_model="m",
    )
    path = (Division(kind="article", number="33"), Division(kind="paragraph", number="1"))
    text = "The controller shall notify the breach within 72 hours."
    return StoredChunk(
        document=document, position=4, chunk=Chunk(kind="text", text=text, path=path)
    )


def test_the_input_numbers_the_facts_with_all_their_wordings() -> None:
    prompt = build_input(_question(), _chunk())

    assert (
        "1. (gdpr) not later than 72 hours after | (edpb-guidelines-2022-09) within 72 hours"
        in prompt
    )
    assert "2. (gdpr) without undue delay" in prompt


def test_the_input_gives_the_question_its_answer_and_the_passage_with_its_location() -> None:
    prompt = build_input(_question(), _chunk())

    assert "<question>\nHow long do I have to report a breach?\n</question>" in prompt
    assert "<answer>\n72 hours (GDPR, Article 33(1)).\n</answer>" in prompt
    assert "<location>GDPR, Article 33(1)</location>" in prompt
    assert prompt.endswith("The controller shall notify the breach within 72 hours.\n</passage>")


def test_the_verdict_is_parsed_from_the_structured_output() -> None:
    judge = OpenAIRelevanceJudge(_FakeOpenAI().client(), "gpt-test")

    verdict = judge.judge(_question(), _chunk())

    assert verdict == Verdict(
        reason="It gives the deadline.", facts=(StatedFact(fact=1, quote="within 72 hours"),)
    )
    assert verdict.relevant


def test_a_passage_stating_no_fact_is_not_relevant() -> None:
    fake = _FakeOpenAI({"reason": "Only the same topic.", "facts": []})

    verdict = OpenAIRelevanceJudge(fake.client(), "gpt-test").judge(_question(), _chunk())

    assert not verdict.relevant


def test_the_request_asks_for_the_verdict_schema_and_keeps_nothing() -> None:
    fake = _FakeOpenAI()

    OpenAIRelevanceJudge(fake.client(), "gpt-test").judge(_question(), _chunk())

    [request] = fake.requests
    assert request["model"] == "gpt-test"
    assert request["instructions"] == INSTRUCTIONS
    assert request["input"] == build_input(_question(), _chunk())
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True


def test_an_incomplete_response_is_an_error() -> None:
    judge = OpenAIRelevanceJudge(_FakeOpenAI(status="incomplete").client(), "gpt-test")

    with pytest.raises(JudgeError, match="incomplete"):
        judge.judge(_question(), _chunk())


def test_a_quote_copied_from_the_chunk_is_supported_despite_case_and_punctuation() -> None:
    verdict = Verdict(reason="", facts=(StatedFact(fact=1, quote="Notify the breach, within 72"),))

    assert unsupported_facts(verdict, _question(), _chunk()) == []


def test_a_quote_not_in_the_chunk_is_unsupported() -> None:
    paraphrase = StatedFact(fact=1, quote="report the breach in 72 hours")
    verdict = Verdict(reason="", facts=(paraphrase,))

    assert unsupported_facts(verdict, _question(), _chunk()) == [paraphrase]


@pytest.mark.parametrize("number", [0, 3])
def test_a_fact_number_the_question_does_not_have_is_unsupported(number: int) -> None:
    stated = StatedFact(fact=number, quote="within 72 hours")

    assert unsupported_facts(Verdict(reason="", facts=(stated,)), _question(), _chunk()) == [stated]
