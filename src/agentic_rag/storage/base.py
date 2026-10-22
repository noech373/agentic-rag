"""Storage interface: what the ingestion writes and what the retrievers search.

The rest of the code depends only on these protocols, so another database is one new class
(see `postgres.py`). The two protocols are separate because the ingestion only writes and a
retriever only searches: a backend without vectors, for instance, can still serve keyword search.
"""

from collections.abc import Collection, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from agentic_rag.document import Chunk


class IndexedDocument(BaseModel):
    """A stored document: how to cite it, who may read it, and what its chunks were built from."""

    model_config = ConfigDict(frozen=True)

    id: str  # the manifest's slug: "gdpr"
    title: str
    short_title: str  # what the answer cites: "GDPR"
    source_url: str | None  # None for internal documents
    regulations: tuple[str, ...]
    groups: tuple[str, ...]  # who may read it: ("public",); empty: nobody
    sha256: str  # of the source file: an unchanged file is not ingested again
    embedding_model: str  # of all its chunks: a query is only compared with vectors of its model


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: IndexedDocument
    position: int  # in the document, 0-based
    chunk: Chunk
    score: float  # higher is more relevant; only comparable within one retriever


class ChunkStore(Protocol):
    """Written by the ingestion, one whole document at a time."""

    def create_schema(self) -> None:
        """Create what the store needs and is missing (tables, indexes...); safe to repeat."""
        ...

    def indexed_documents(self) -> dict[str, IndexedDocument]: ...

    def replace_document(
        self,
        document: IndexedDocument,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store the document and its chunks in place of any previous version, atomically."""
        ...

    def delete_documents(self, ids: Collection[str]) -> None: ...


class VectorSearch(Protocol):
    def vector_search(
        self, embedding: Sequence[float], *, embedding_model: str, k: int
    ) -> list[RetrievedChunk]:
        """The `k` chunks nearest to `embedding` among those of `embedding_model`, best first."""
        ...
