from collections.abc import Sequence

import pytest

from agentic_rag.document import Chunk, IndexedDocument, RetrievedChunk
from agentic_rag.retrieval.base import Retriever
from agentic_rag.retrieval.reranking import CANDIDATES, RerankingRetriever


def _result(name: str, score: float = 1.0) -> RetrievedChunk:
    """A chunk named "document#position", whose text is its name."""
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
    return RetrievedChunk(document=document, position=int(position), chunk=chunk, score=score)


def _names(results: list[RetrievedChunk]) -> list[str]:
    return [f"{result.document.id}#{result.position}" for result in results]


class _FakeRetriever:
    """Returns its ranking, cut at `k`, and remembers the `k` it was asked for."""

    def __init__(self, *names: str) -> None:
        self._ranking = [_result(name, score=100.0 - rank) for rank, name in enumerate(names)]
        self.asked: list[int] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.asked.append(k)
        return self._ranking[:k]


class _FakeReranker:
    """Scores each text from a table (0 by default) and remembers what it was given."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self._scores = scores or {}
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return [self._scores.get(text, 0.0) for text in texts]


def test_a_reranking_retriever_is_a_retriever() -> None:
    retriever: Retriever = RerankingRetriever(_FakeRetriever("gdpr#1"), _FakeReranker())

    assert _names(retriever.retrieve("question", 1)) == ["gdpr#1"]


def test_chunks_come_back_in_the_order_of_the_reranker() -> None:
    """The real case: the chunk the retriever ranks last is the one the reranker prefers."""
    inner = _FakeRetriever("gdpr#66", "gdpr#68", "gdpr#222")
    reranker = _FakeReranker({"gdpr#66": 0.2, "gdpr#68": 0.1, "gdpr#222": 0.9})

    results = RerankingRetriever(inner, reranker).retrieve("question", 3)

    assert _names(results) == ["gdpr#222", "gdpr#66", "gdpr#68"]


def test_only_the_k_best_come_back() -> None:
    inner = _FakeRetriever("gdpr#1", "gdpr#2", "gdpr#3")
    reranker = _FakeReranker({"gdpr#1": 0.1, "gdpr#2": 0.2, "gdpr#3": 0.3})

    results = RerankingRetriever(inner, reranker).retrieve("question", 2)

    assert _names(results) == ["gdpr#3", "gdpr#2"]


def test_a_small_k_still_asks_for_all_the_candidates() -> None:
    inner = _FakeRetriever("gdpr#1")

    RerankingRetriever(inner, _FakeReranker()).retrieve("question", 8)

    assert inner.asked == [CANDIDATES]


def test_a_k_above_the_candidates_asks_for_k() -> None:
    inner = _FakeRetriever("gdpr#1")

    RerankingRetriever(inner, _FakeReranker()).retrieve("question", CANDIDATES + 10)

    assert inner.asked == [CANDIDATES + 10]


def test_there_are_50_candidates() -> None:
    """The ceiling measured before coding (recall@50) holds for this number."""
    assert CANDIDATES == 50


def test_the_reranker_reads_the_question_and_the_text_of_every_candidate() -> None:
    reranker = _FakeReranker()
    inner = _FakeRetriever("gdpr#1", "gdpr#2", "gdpr#3")

    RerankingRetriever(inner, reranker).retrieve("Can we dismiss our DPO?", 1)

    assert reranker.calls == [("Can we dismiss our DPO?", ["gdpr#1", "gdpr#2", "gdpr#3"])]


def test_a_chunk_carries_the_score_of_the_reranker() -> None:
    inner = _FakeRetriever("gdpr#1", "gdpr#2")
    reranker = _FakeReranker({"gdpr#1": 0.25, "gdpr#2": 0.75})

    results = RerankingRetriever(inner, reranker).retrieve("question", 2)

    assert [result.score for result in results] == pytest.approx([0.75, 0.25])


def test_only_the_score_of_a_chunk_changes() -> None:
    inner = _FakeRetriever("gdpr#7")
    before = inner.retrieve("question", 1)[0]

    after = RerankingRetriever(inner, _FakeReranker({"gdpr#7": 0.5})).retrieve("question", 1)[0]

    assert after == before.model_copy(update={"score": 0.5})


def test_ties_keep_the_order_of_the_retriever() -> None:
    """So that the same scores always give the same ranking."""
    inner = _FakeRetriever("gdpr#3", "gdpr#1", "gdpr#2")

    results = RerankingRetriever(inner, _FakeReranker()).retrieve("question", 3)

    assert _names(results) == ["gdpr#3", "gdpr#1", "gdpr#2"]


def test_the_top_8_is_the_start_of_the_top_50() -> None:
    """The evaluation asks for 50 and reads recall@8 from their start: the RAG must get the same."""
    names = [f"gdpr#{position}" for position in range(60)]
    scores = {name: float(position % 7) for position, name in enumerate(names)}

    top_8 = RerankingRetriever(_FakeRetriever(*names), _FakeReranker(scores)).retrieve("q", 8)
    top_50 = RerankingRetriever(_FakeRetriever(*names), _FakeReranker(scores)).retrieve("q", 50)

    assert top_8 == top_50[:8]


def test_a_missing_score_is_an_error() -> None:
    """Otherwise the chunks without a score would silently vanish from the ranking."""

    class _ShortReranker:
        def score(self, query: str, texts: Sequence[str]) -> list[float]:
            return [0.5]

    retriever = RerankingRetriever(_FakeRetriever("gdpr#1", "gdpr#2"), _ShortReranker())

    with pytest.raises(ValueError, match="zip"):
        retriever.retrieve("question", 2)


def test_no_candidate_gives_no_result() -> None:
    results = RerankingRetriever(_FakeRetriever(), _FakeReranker()).retrieve("question", 8)

    assert results == []
