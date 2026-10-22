"""Postgres schema: the ingested documents and their chunks, with embeddings.

The tables are designed from the queries they serve: skip a document whose file has not changed
(`sha256`), delete all its chunks when it has (`ON DELETE CASCADE`), keep only the documents a
user may see (`groups`), and cite a chunk ("GDPR Art. 33(1)", with a link) from `path`.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Fixed by the schema, not read from the settings: a vector column has one dimension, and changing
# the embedding model means a migration. Must match OPENAI_EMBEDDING_DIMENSIONS.
EMBEDDING_DIMENSIONS = 1536

BM25_INDEX = "chunks_text_bm25_idx"  # queries name it: text <@> to_bm25query(:query, BM25_INDEX)


class Base(DeclarativeBase):
    type_annotation_map = {str: Text}  # noqa: RUF012 (SQLAlchemy reads it as a class attribute)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True)  # the manifest's slug: "gdpr"
    title: Mapped[str]
    short_title: Mapped[str]  # what the answer cites: "GDPR"
    source_url: Mapped[str | None]
    regulations: Mapped[list[str]] = mapped_column(ARRAY(Text))
    sha256: Mapped[str]
    groups: Mapped[list[str]] = mapped_column(ARRAY(Text))

    chunks: Mapped[list[Chunk]] = relationship(
        order_by="Chunk.position", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"
    # No vector index: on ~2000 chunks an exact scan takes ~9 ms, while HNSW (approximate) needs
    # hnsw.ef_search = 200 to find 99 % of the true top 20 (measured on random vectors).
    __table_args__ = (
        Index(
            BM25_INDEX, "text", postgresql_using="bm25", postgresql_with={"text_config": "english"}
        ),
    )

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(primary_key=True)  # 0-based, in document order
    kind: Mapped[str]  # "text" or "table"
    text: Mapped[str]
    path: Mapped[list[dict[str, str | None]]] = mapped_column(JSONB)
    page: Mapped[int | None]
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    embedding_model: Mapped[str]
