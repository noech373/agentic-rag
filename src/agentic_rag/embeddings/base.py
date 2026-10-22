"""Embedding models: texts to vectors, for the vector search.

The rest of the code depends only on `Embedder`: another model or provider is one new class.
"""

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """The vector space, stored with the vectors: "text-embedding-3-small@1536"."""
        ...

    # Two methods, as in LangChain: some models (e5, Cohere, Voyage) embed a query differently
    # from the documents it should match.
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
