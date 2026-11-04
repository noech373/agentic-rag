"""Lexical retrieval (BM25): the chunks sharing the most words with the question,
rare ones first.
"""

from agentic_rag.document import RetrievedChunk
from agentic_rag.retrieval.base import Retriever
from agentic_rag.storage.base import KeywordSearch


class BM25Retriever(Retriever):
    def __init__(self, search: KeywordSearch) -> None:
        self._search = search

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._search.keyword_search(query, k=k)
