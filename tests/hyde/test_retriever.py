from agentic_rag.document import Chunk, IndexedDocument, RetrievedChunk
from agentic_rag.hyde.retriever import HydeRetriever
from agentic_rag.retrieval.base import Retriever

QUESTION = "Our privacy person keeps blocking projects. Can we let her go for it?"
ANSWER = "Under the GDPR, the data protection officer shall not be dismissed for performing tasks."


def _result(name: str) -> RetrievedChunk:
    document_id, position = name.split("#")
    document = IndexedDocument(
        id=document_id,
        title=document_id,
        short_title=document_id.upper(),
        source_url=None,
        regulations=(),
        groups=("public",),
        sha256="0" * 64,
        embedding_model="fake@2",
    )
    chunk = Chunk(kind="text", text=name, path=())
    return RetrievedChunk(document=document, position=int(position), chunk=chunk, score=1.0)


class _FakeRetriever:
    """Returns its ranking, cut at `k`, and remembers each query and `k` it was given."""

    def __init__(self, *names: str) -> None:
        self._ranking = [_result(name) for name in names]
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self._ranking[:k]


class _FakeWriter:
    """Answers every question with the same text, and remembers the questions."""

    name = "fake"

    def __init__(self, answer: str = ANSWER) -> None:
        self._answer = answer
        self.questions: list[str] = []

    def write(self, question: str) -> str:
        self.questions.append(question)
        return self._answer


def test_a_hyde_retriever_is_a_retriever() -> None:
    retriever: Retriever = HydeRetriever(_FakeRetriever("gdpr#1"), _FakeWriter())

    assert retriever.retrieve(QUESTION, 1) == [_result("gdpr#1")]


def test_the_writer_reads_the_question() -> None:
    writer = _FakeWriter()

    HydeRetriever(_FakeRetriever(), writer).retrieve(QUESTION, 8)

    assert writer.questions == [QUESTION]


def test_the_search_uses_the_answer_alone() -> None:
    """Pure HyDE: the question's own words are not added to the search."""
    inner = _FakeRetriever()

    HydeRetriever(inner, _FakeWriter()).retrieve(QUESTION, 8)

    assert inner.calls == [(ANSWER, 8)]


def test_the_chunks_of_the_inner_retriever_come_back_unchanged() -> None:
    inner = _FakeRetriever("gdpr#68", "gdpr#204", "gdpr#96")

    results = HydeRetriever(inner, _FakeWriter()).retrieve(QUESTION, 2)

    assert results == [_result("gdpr#68"), _result("gdpr#204")]


def test_one_answer_per_search() -> None:
    writer = _FakeWriter()
    retriever = HydeRetriever(_FakeRetriever(), writer)

    retriever.retrieve(QUESTION, 8)
    retriever.retrieve("Who must test the ICT business continuity plans?", 8)

    assert writer.questions == [QUESTION, "Who must test the ICT business continuity plans?"]
