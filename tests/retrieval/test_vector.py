from collections.abc import Sequence

from agentic_rag.document import Chunk
from agentic_rag.embeddings.base import Embedder
from agentic_rag.retrieval.vector import VectorRetriever
from agentic_rag.storage.base import IndexedDocument, RetrievedChunk, VectorSearch


class _FakeEmbedder(Embedder):
    @property
    def name(self) -> str:
        return "fake@2"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0]


class _FakeSearch(VectorSearch):
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[list[float], str, int]] = []

    def vector_search(
        self, embedding: Sequence[float], *, embedding_model: str, k: int
    ) -> list[RetrievedChunk]:
        self.calls.append((list(embedding), embedding_model, k))
        return self.results[:k]


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


def test_the_query_is_embedded_and_searched_among_vectors_of_its_model() -> None:
    search = _FakeSearch([_result(0), _result(1), _result(2)])

    results = VectorRetriever(search, _FakeEmbedder()).retrieve("breach", k=2)

    assert search.calls == [([6.0, 0.0], "fake@2", 2)]  # embed_query, not embed_documents
    assert results == [_result(0), _result(1)]
