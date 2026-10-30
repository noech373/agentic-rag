"""Retrievers: from a question to the chunks most likely to answer it.

The RAG depends only on `Retriever`: BM25, hybrid search or a reranker is one new class.
"""

from typing import Protocol

from agentic_rag.storage.base import RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """At most `k` chunks, the most relevant to `query`, best first: fewer if fewer
        are relevant."""
        ...
