"""Reranking: a finer model reorders the candidates of another retriever."""

from agentic_rag.document import RetrievedChunk
from agentic_rag.reranking.base import Reranker
from agentic_rag.retrieval.base import Retriever

CANDIDATES = 50  # Whatever k: the top 8 is then the start of the top 50


class RerankingRetriever(Retriever):
    def __init__(self, retriever: Retriever, reranker: Reranker) -> None:
        self._reranker = reranker
        self._retriever = retriever

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """The `k` best candidates by the reranker's scores; ties keep the retriever's order."""
        candidates = self._retriever.retrieve(query, max(CANDIDATES, k))
        texts = [candidate.chunk.text for candidate in candidates]
        scores = self._reranker.score(query, texts)

        ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)[:k]

        return [pair[0].model_copy(update={"score": pair[1]}) for pair in ranked]
