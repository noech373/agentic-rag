"""HyDE (Gao et al., 2022): search with a hypothetical answer instead of the question."""

from agentic_rag.document import RetrievedChunk
from agentic_rag.hyde.base import AnswerWriter
from agentic_rag.retrieval.base import Retriever


class HydeRetriever(Retriever):
    def __init__(self, retriever: Retriever, writer: AnswerWriter) -> None:
        self._retriever = retriever
        self._writer = writer

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """The `k` chunks closest to the answer the writer gives to `query`."""
        return self._retriever.retrieve(self._writer.write(query), k)
