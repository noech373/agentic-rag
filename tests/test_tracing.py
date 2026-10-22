import json
from collections.abc import Sequence
from typing import Any

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agentic_rag.document import Chunk, Division
from agentic_rag.generation.base import NO_ANSWER, Answer, GenerationError, Generator
from agentic_rag.rag import RAG
from agentic_rag.retrieval.base import Retriever
from agentic_rag.storage.base import IndexedDocument, RetrievedChunk
from agentic_rag.tracing import TracedRAG, TracedRetriever

_EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module")
def langfuse() -> Langfuse:
    """A real client whose spans stay in memory: nothing is sent."""
    return Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",  # noqa: S106
        base_url="http://127.0.0.1:9",  # unreachable, in case something tried
        span_exporter=_EXPORTER,
    )


@pytest.fixture(autouse=True)
def clear_exporter() -> None:
    _EXPORTER.clear()


class _FakeRetriever(Retriever):
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return self.results[:k]


class _FakeGenerator(Generator):
    def __init__(self, text: str = "Within 72 hours [1].", error: bool = False) -> None:
        self.text = text
        self.error = error

    def generate(self, question: str, sources: Sequence[RetrievedChunk]) -> Answer:
        if self.error:
            raise GenerationError("response incomplete")
        return Answer(text=self.text, sources=tuple(sources))


def _result(article: str, score: float) -> RetrievedChunk:
    document = IndexedDocument(
        id="gdpr",
        title="General Data Protection Regulation",
        short_title="GDPR",
        source_url=None,
        regulations=("GDPR",),
        groups=("public",),
        sha256="0" * 64,
        embedding_model="fake@2",
    )
    path = (Division(kind="article", number=article, title="Notification"),)
    chunk = Chunk(kind="text", text=f"Text of Article {article}.", path=path)
    return RetrievedChunk(document=document, position=0, chunk=chunk, score=score)


_RESULTS = [_result("33", 0.61234567), _result("34", 0.5)]


def _rag(langfuse: Langfuse, generator: Generator | None = None, k: int = 8) -> TracedRAG:
    retriever = TracedRetriever(_FakeRetriever(_RESULTS), langfuse)
    return TracedRAG(retriever, generator or _FakeGenerator(), langfuse, k=k)


def _spans(langfuse: Langfuse) -> dict[str, ReadableSpan]:
    langfuse.flush()
    return {span.name: span for span in _EXPORTER.get_finished_spans()}


def _attribute(span: ReadableSpan, name: str) -> Any:
    value = (span.attributes or {})[f"langfuse.observation.{name}"]
    return json.loads(value) if isinstance(value, str) and value[:1] in "[{" else value


def test_traced_rag_answers_like_the_rag(langfuse: Langfuse) -> None:
    traced = _rag(langfuse).ask("How fast?")
    plain = RAG(_FakeRetriever(_RESULTS), _FakeGenerator()).ask("How fast?")

    assert traced == plain


def test_a_question_is_one_trace_whose_root_is_the_question_and_its_answer(
    langfuse: Langfuse,
) -> None:
    _rag(langfuse, k=2).ask("How fast?")

    spans = _spans(langfuse)
    ask = spans["ask"]
    assert ask.parent is None
    assert _attribute(ask, "type") == "chain"
    assert _attribute(ask, "input") == "How fast?"
    assert _attribute(ask, "output") == {
        "text": "Within 72 hours [1].",
        "cited": {"1": "GDPR, Article 33"},
    }
    assert _attribute(ask, "metadata.k") == "2"


def test_the_retrieval_is_nested_with_what_it_found_and_the_scores(langfuse: Langfuse) -> None:
    _rag(langfuse).ask("How fast?")

    spans = _spans(langfuse)
    retrieve = spans["retrieve"]
    assert retrieve.parent == spans["ask"].context  # same trace, child of the question
    assert _attribute(retrieve, "type") == "retriever"
    assert _attribute(retrieve, "input") == {"query": "How fast?", "k": 8}
    assert _attribute(retrieve, "output") == [
        {"citation": "GDPR, Article 33", "score": 0.6123, "text": "Text of Article 33."},
        {"citation": "GDPR, Article 34", "score": 0.5, "text": "Text of Article 34."},
    ]


def test_a_retriever_alone_is_its_own_trace(langfuse: Langfuse) -> None:
    TracedRetriever(_FakeRetriever(_RESULTS), langfuse).retrieve("How fast?", 1)

    spans = _spans(langfuse)
    assert list(spans) == ["retrieve"]
    assert len(_attribute(spans["retrieve"], "output")) == 1


def test_nothing_retrieved_is_traced_with_the_no_answer(langfuse: Langfuse) -> None:
    retriever = TracedRetriever(_FakeRetriever([]), langfuse)
    TracedRAG(retriever, _FakeGenerator(), langfuse).ask("How fast?")

    spans = _spans(langfuse)
    assert _attribute(spans["retrieve"], "output") == []
    assert _attribute(spans["ask"], "output") == {"text": NO_ANSWER, "cited": {}}


def test_a_failure_is_raised_and_recorded_as_an_error(langfuse: Langfuse) -> None:
    with pytest.raises(GenerationError):
        _rag(langfuse, generator=_FakeGenerator(error=True)).ask("How fast?")

    ask = _spans(langfuse)["ask"]
    assert ask.status.status_code is StatusCode.ERROR
    assert ask.status.description == "GenerationError: response incomplete"
    assert [event.name for event in ask.events] == ["exception"]
