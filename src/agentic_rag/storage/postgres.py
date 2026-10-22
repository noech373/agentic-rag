"""Postgres backend: the schema and the store (pgvector for vectors, pg_textsearch for BM25).

The tables are designed from the queries they serve: skip a document whose file has not changed
(`sha256`), delete all its chunks when it has (`ON DELETE CASCADE`), keep only the documents a
user may see (`groups`), and cite a chunk ("GDPR Art. 33(1)", with a link) from `path`.
"""

from collections.abc import Collection, Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import URL, Engine, ForeignKey, Index, Text, delete, insert, select, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from agentic_rag.config import PostgresSettings
from agentic_rag.document import Chunk
from agentic_rag.storage.base import ChunkStore, IndexedDocument, RetrievedChunk, VectorSearch

BM25_INDEX = "chunks_text_bm25_idx"  # queries name it: text <@> to_bm25query(:query, BM25_INDEX)

_EXTENSIONS = ("vector", "pg_textsearch")


class Base(DeclarativeBase):
    type_annotation_map = {str: Text}  # noqa: RUF012 (SQLAlchemy reads it as a class attribute)


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str]
    short_title: Mapped[str]
    source_url: Mapped[str | None]
    regulations: Mapped[list[str]] = mapped_column(ARRAY(Text))
    groups: Mapped[list[str]] = mapped_column(ARRAY(Text))  # no default: forgetting it fails
    sha256: Mapped[str]
    embedding_model: Mapped[str]


class ChunkRow(Base):
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
    path: Mapped[list[dict[str, str | None]]] = mapped_column(JSONB)  # the Divisions, dumped
    page: Mapped[int | None]
    # No dimension, so any embedding model fits: searches only compare the vectors of one model.
    # Deferred: written and compared in SQL, never read back by the application.
    embedding: Mapped[list[float]] = mapped_column(Vector(), deferred=True)


def database_url(settings: PostgresSettings, database: str | None = None) -> URL:
    return URL.create(
        "postgresql+psycopg",
        username=settings.user,
        password=settings.password.get_secret_value(),
        host=settings.host,
        port=settings.port,
        database=settings.db if database is None else database,
    )


class PostgresStore(ChunkStore, VectorSearch):
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._sessions = sessionmaker(engine)

    def create_schema(self) -> None:
        """Create the missing extensions, tables and indexes. Existing tables are left as they are.

        After a schema change, `drop_schema` and ingest again: the corpus rebuilds the database in
        a minute. Migrations (Alembic) will come once some data cannot be rebuilt.
        """
        with self._engine.begin() as connection:
            for extension in _EXTENSIONS:
                connection.execute(text(f"CREATE EXTENSION IF NOT EXISTS {extension}"))
        Base.metadata.create_all(self._engine)

    def drop_schema(self) -> None:
        Base.metadata.drop_all(self._engine)

    def indexed_documents(self) -> dict[str, IndexedDocument]:
        with self._sessions() as session:
            return {row.id: _document(row) for row in session.scalars(select(DocumentRow))}

    def replace_document(
        self,
        document: IndexedDocument,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        rows = [
            {
                **chunk.model_dump(mode="json"),
                "document_id": document.id,
                "position": position,
                "embedding": list(embedding),
            }
            for position, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        with self._sessions.begin() as session:
            session.execute(delete(DocumentRow).where(DocumentRow.id == document.id))  # cascades
            session.execute(insert(DocumentRow).values(**document.model_dump(mode="json")))
            if rows:
                session.execute(insert(ChunkRow), rows)

    def delete_documents(self, ids: Collection[str]) -> None:
        with self._sessions.begin() as session:
            session.execute(delete(DocumentRow).where(DocumentRow.id.in_(ids)))

    def vector_search(
        self, embedding: Sequence[float], *, embedding_model: str, k: int
    ) -> list[RetrievedChunk]:
        distance = ChunkRow.embedding.cosine_distance(list(embedding))
        statement = (
            select(ChunkRow, DocumentRow, distance)
            .join(DocumentRow)
            .where(DocumentRow.embedding_model == embedding_model)
            .order_by(distance, ChunkRow.document_id, ChunkRow.position)
            .limit(k)
        )
        with self._sessions() as session:
            return [
                RetrievedChunk(
                    document=_document(document),
                    position=chunk.position,
                    chunk=Chunk.model_validate(chunk, from_attributes=True),
                    score=1 - distance,  # cosine similarity
                )
                for chunk, document, distance in session.execute(statement)
            ]


def _document(row: DocumentRow) -> IndexedDocument:
    return IndexedDocument.model_validate(row, from_attributes=True)
