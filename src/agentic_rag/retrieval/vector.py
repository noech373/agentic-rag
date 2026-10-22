"""Dense retrieval: the chunks whose embedding is nearest to the question's."""

from agentic_rag.embeddings.base import Embedder
from agentic_rag.retrieval.base import Retriever
from agentic_rag.storage.base import RetrievedChunk, VectorSearch


class VectorRetriever(Retriever):
    def __init__(self, search: VectorSearch, embedder: Embedder) -> None:
        self._search = search
        self._embedder = embedder

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        embedding = self._embedder.embed_query(query)
        return self._search.vector_search(embedding, embedding_model=self._embedder.name, k=k)
