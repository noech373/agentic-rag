"""Hybrid retrieval: rankings of several retrievers fused into one by their ranks (RRF)."""

from collections.abc import Sequence

from agentic_rag.retrieval.base import Retriever
from agentic_rag.storage.base import RetrievedChunk

RRF_K = 60  # Cormack et al., 2009: a first rank weighs little more than a second one
CANDIDATES = 50  # Per retriever, whatever k: the top 8 is then the start of the top 50


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], *, k: int
) -> list[RetrievedChunk]:
    """The `k` best chunks by the sum of 1 / (RRF_K + rank) over the rankings that have them.

    Only ranks count: the retrievers' scores are on scales that cannot be compared. Ties keep the
    order in which chunks were first seen, so the same rankings always fuse in the same order.
    """
    scores: dict[tuple[str, int], float] = {}
    chunks: dict[tuple[str, int], RetrievedChunk] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            key = (result.document.id, result.position)
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)
            chunks.setdefault(key, result)

    best = sorted(scores, key=lambda key: scores[key], reverse=True)[:k]
    return [chunks[key].model_copy(update={"score": scores[key]}) for key in best]


class HybridRetriever(Retriever):
    def __init__(self, retrievers: Sequence[Retriever]) -> None:
        self._retrievers = retrievers

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        rankings = [retriever.retrieve(query, max(k, CANDIDATES)) for retriever in self._retrievers]

        return reciprocal_rank_fusion(rankings, k=k)
