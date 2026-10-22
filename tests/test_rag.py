from collections.abc import Sequence

from agentic_rag.document import Chunk
from agentic_rag.generation.base import NO_ANSWER, Answer, Generator
from agentic_rag.rag import RAG
from agentic_rag.retrieval.base import Retriever
from agentic_rag.storage.base import IndexedDocument, RetrievedChunk


class _FakeRetriever(Retriever):
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self.results[:k]


class _FakeGenerator(Generator):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(self, question: str, sources: Sequence[RetrievedChunk]) -> Answer:
        self.calls.append((question, list(sources)))
        return Answer(text="Within 72 hours [1].", sources=tuple(sources))


def _result(position: int) -> RetrievedChunk:
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
    chunk = Chunk(kind="text", text=f"Chunk {position}.", path=())
    return RetrievedChunk(document=document, position=position, chunk=chunk, score=1.0)


def test_the_k_retrieved_sources_go_to_the_generator_with_the_question() -> None:
    retriever = _FakeRetriever([_result(i) for i in range(5)])
    generator = _FakeGenerator()

    answer = RAG(retriever, generator, k=3).ask("Breach deadline?")

    assert retriever.calls == [("Breach deadline?", 3)]
    assert generator.calls == [("Breach deadline?", [_result(0), _result(1), _result(2)])]
    assert answer.text == "Within 72 hours [1]."


def test_without_any_source_the_answer_is_no_answer_and_the_llm_is_not_called() -> None:
    generator = _FakeGenerator()

    answer = RAG(_FakeRetriever([]), generator).ask("Breach deadline?")

    assert answer == Answer(text=NO_ANSWER, sources=())
    assert generator.calls == []
