import pytest

from agentic_rag.document import Chunk
from agentic_rag.retrieval.base import Retriever
from agentic_rag.retrieval.hybrid import CANDIDATES, HybridRetriever, reciprocal_rank_fusion
from agentic_rag.storage.base import IndexedDocument, RetrievedChunk


def _result(name: str, score: float = 1.0) -> RetrievedChunk:
    """A chunk named "document#position", with the score its retriever gave it."""
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
    chunk = Chunk(kind="text", text=f"Chunk {name}.", path=())
    return RetrievedChunk(document=document, position=int(position), chunk=chunk, score=score)


def _ranking(*names: str) -> list[RetrievedChunk]:
    """One retriever's results, best first, with scores on its own scale."""
    return [_result(name, score=100.0 - rank) for rank, name in enumerate(names)]


def _names(results: list[RetrievedChunk]) -> list[str]:
    return [f"{result.document.id}#{result.position}" for result in results]


def test_the_score_is_the_sum_of_one_over_60_plus_the_rank() -> None:
    fused = reciprocal_rank_fusion([_ranking("dora#1", "gdpr#2"), _ranking("gdpr#2")], k=10)

    assert _names(fused) == ["gdpr#2", "dora#1"]
    assert [result.score for result in fused] == pytest.approx([1 / 62 + 1 / 61, 1 / 61])


def test_second_in_both_rankings_beats_first_in_only_one() -> None:
    """The point of RRF: what both retrievers find comes first, whatever their scores."""
    fused = reciprocal_rank_fusion(
        [_ranking("dora#158", "gdpr#33"), _ranking("nis2#23", "gdpr#33")], k=10
    )

    assert _names(fused)[0] == "gdpr#33"


def test_only_ranks_count_not_the_retrievers_scores() -> None:
    scaled = [_result("dora#1", score=1000.0), _result("gdpr#2", score=999.0)]
    fused = reciprocal_rank_fusion([_ranking("gdpr#2", "dora#1"), scaled], k=10)

    assert [result.score for result in fused] == pytest.approx([1 / 61 + 1 / 62] * 2)


def test_a_chunk_is_the_same_whatever_its_score_in_each_ranking() -> None:
    """Chunks are matched by document and position: each retriever gives its own score."""
    fused = reciprocal_rank_fusion(
        [[_result("gdpr#2", score=0.8)], [_result("gdpr#2", score=12.5)]], k=10
    )

    assert _names(fused) == ["gdpr#2"]


def test_chunks_found_by_one_retriever_only_are_kept() -> None:
    fused = reciprocal_rank_fusion([_ranking("dora#1"), _ranking("nis2#7")], k=10)

    assert sorted(_names(fused)) == ["dora#1", "nis2#7"]


def test_ties_keep_the_order_of_the_rankings() -> None:
    """Equal scores in a stable order: two runs of the same evaluation retrieve the same chunks."""
    fused = reciprocal_rank_fusion([_ranking("nis2#7"), _ranking("dora#1")], k=10)

    assert _names(fused) == ["nis2#7", "dora#1"]


def test_at_most_k_chunks_are_kept() -> None:
    fused = reciprocal_rank_fusion([_ranking("a#0", "a#1", "a#2"), _ranking("b#0", "b#1")], k=3)

    assert len(fused) == 3


def test_the_chunks_and_documents_are_kept_unchanged() -> None:
    [fused] = reciprocal_rank_fusion([[_result("gdpr#2")]], k=10)

    assert fused.document == _result("gdpr#2").document
    assert fused.chunk == _result("gdpr#2").chunk


def test_no_rankings_or_empty_ones_fuse_to_nothing() -> None:
    assert reciprocal_rank_fusion([], k=10) == []
    assert reciprocal_rank_fusion([[], []], k=10) == []


# HybridRetriever


class _FakeRetriever(Retriever):
    """A retriever with a fixed ranking, which records what it was asked."""

    def __init__(self, *names: str) -> None:
        self.ranking = _ranking(*names)
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self.ranking[:k]


def _long_ranking(document_id: str, length: int) -> list[str]:
    return [f"{document_id}#{position}" for position in range(length)]


def test_the_retriever_asks_each_retriever_the_query_as_is() -> None:
    vector, bm25 = _FakeRetriever("gdpr#2"), _FakeRetriever("dora#1")

    HybridRetriever([vector, bm25]).retrieve("Personal data breach?", k=8)

    assert [query for query, _ in vector.calls + bm25.calls] == ["Personal data breach?"] * 2


def test_the_retriever_asks_for_the_same_candidates_whatever_k() -> None:
    vector, bm25 = _FakeRetriever("gdpr#2"), _FakeRetriever("dora#1")
    hybrid = HybridRetriever([vector, bm25])

    hybrid.retrieve("breach", k=8)
    hybrid.retrieve("breach", k=50)

    assert [k for _, k in vector.calls] == [CANDIDATES, CANDIDATES]
    assert [k for _, k in bm25.calls] == [CANDIDATES, CANDIDATES]


def test_the_retriever_asks_for_k_candidates_when_k_is_larger() -> None:
    vector = _FakeRetriever("gdpr#2")

    HybridRetriever([vector]).retrieve("breach", k=CANDIDATES + 10)

    assert vector.calls == [("breach", CANDIDATES + 10)]


def test_the_retriever_fuses_the_rankings_by_rrf() -> None:
    vector = _FakeRetriever("dora#158", "gdpr#33")
    bm25 = _FakeRetriever("nis2#23", "gdpr#33")

    results = HybridRetriever([vector, bm25]).retrieve("breach", k=10)

    assert results == reciprocal_rank_fusion([vector.ranking, bm25.ranking], k=10)
    assert _names(results)[0] == "gdpr#33"


def test_the_retriever_returns_at_most_k_chunks() -> None:
    vector = _FakeRetriever(*_long_ranking("gdpr", 20))
    bm25 = _FakeRetriever(*_long_ranking("dora", 20))

    assert len(HybridRetriever([vector, bm25]).retrieve("breach", k=8)) == 8


def test_the_retriever_top_8_is_the_start_of_its_top_50() -> None:
    """What the evaluation measures (top 8 of 50) is what `ask` returns (k=8).

    gdpr#9 is 10th for the vector search and 1st for BM25: it is among the best
    only if the vector search was asked for more than 8 candidates.
    """
    vector = _FakeRetriever(*_long_ranking("gdpr", 60))
    bm25 = _FakeRetriever("gdpr#9", *_long_ranking("dora", 59))
    hybrid = HybridRetriever([vector, bm25])

    top_8 = hybrid.retrieve("breach", k=8)

    assert top_8 == hybrid.retrieve("breach", k=50)[:8]
    assert _names(top_8)[0] == "gdpr#9"


def test_the_retriever_breaks_ties_in_favour_of_the_first_retriever() -> None:
    vector, bm25 = _FakeRetriever("gdpr#33"), _FakeRetriever("dora#158")

    assert _names(HybridRetriever([vector, bm25]).retrieve("breach", k=2)) == [
        "gdpr#33",
        "dora#158",
    ]
    assert _names(HybridRetriever([bm25, vector]).retrieve("breach", k=2)) == [
        "dora#158",
        "gdpr#33",
    ]
