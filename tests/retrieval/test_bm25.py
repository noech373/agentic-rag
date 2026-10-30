from agentic_rag.document import Chunk
from agentic_rag.retrieval.bm25 import BM25Retriever
from agentic_rag.storage.base import IndexedDocument, KeywordSearch, RetrievedChunk


class _FakeSearch(KeywordSearch):
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def keyword_search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
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


def test_the_query_is_searched_as_is_by_keywords() -> None:
    search = _FakeSearch([_result(0), _result(1), _result(2)])

    results = BM25Retriever(search).retrieve("Personal data breach?", k=2)

    assert search.calls == [("Personal data breach?", 2)]
    assert results == [_result(0), _result(1)]


def test_fewer_than_k_chunks_when_few_share_a_term() -> None:
    search = _FakeSearch([_result(0)])

    assert BM25Retriever(search).retrieve("breach", k=8) == [_result(0)]
