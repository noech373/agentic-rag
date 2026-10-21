from collections.abc import Iterator

import pytest
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import URL, Engine, create_engine, delete, func, select, text
from sqlalchemy.exc import DataError, IntegrityError, OperationalError
from sqlalchemy.orm import Session

from agentic_rag.config import OpenAISettings, PostgresSettings
from agentic_rag.db import EMBEDDING_DIMENSIONS, Base, Chunk, Document
from agentic_rag.ingestion.document import Division

_ARTICLE_33 = (
    Division(kind="chapter", number="IV", title="Controller and processor"),
    Division(kind="section", number="2", title="Security of personal data"),
    Division(kind="article", number="33", title="Notification of a personal data breach"),
)


def test_embedding_dimensions_match_the_default_settings() -> None:
    assert OpenAISettings.model_fields["embedding_dimensions"].default == EMBEDDING_DIMENSIONS


# --- On a real Postgres (skipped until `docker compose up -d` has been run) ---

_TEST_DATABASE = "agentic_rag_test"  # created, then dropped, on the server of the .env settings


def _url(settings: PostgresSettings, database: str) -> URL:
    return URL.create(
        "postgresql+psycopg",
        username=settings.user,
        password=settings.password.get_secret_value(),
        host=settings.host,
        port=settings.port,
        database=database,
    )


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    """A fresh database with the schema, dropped after the tests."""
    try:
        settings = PostgresSettings()  # pyright: ignore[reportCallIssue]  (values come from env)
    except ValidationError:
        pytest.skip("POSTGRES_* settings not found: cp .env.example .env")
    # CREATE DATABASE cannot run inside a transaction, hence AUTOCOMMIT.
    admin = create_engine(
        _url(settings, settings.db),
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 3},
    )
    try:
        with admin.connect() as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS {_TEST_DATABASE} WITH (FORCE)"))
            connection.execute(text(f"CREATE DATABASE {_TEST_DATABASE}"))
    except OperationalError:
        admin.dispose()
        pytest.skip("Postgres not reachable: run `docker compose up -d`")

    engine = create_engine(_url(settings, _TEST_DATABASE))
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION vector"))
    Base.metadata.create_all(engine)
    yield engine

    engine.dispose()
    with admin.connect() as connection:
        connection.execute(text(f"DROP DATABASE {_TEST_DATABASE} WITH (FORCE)"))
    admin.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE documents, chunks"))
    with Session(engine) as session:
        yield session


def _document(id_: str = "gdpr", groups: list[str] | None = None) -> Document:
    return Document(
        id=id_,
        title=f"Title of {id_}",
        short_title=id_.upper(),
        source_url=f"https://example.org/{id_}",
        regulations=["GDPR"],
        sha256="0" * 64,
        groups=["public"] if groups is None else groups,
    )


def _vector(*head: float) -> list[float]:
    """An embedding: the given first values, then zeros."""
    return [*head, *[0.0] * (EMBEDDING_DIMENSIONS - len(head))]


def _chunk(
    document_id: str | None = "gdpr",  # None: set from the document it is added to
    position: int = 0,
    path: tuple[Division, ...] = _ARTICLE_33,
    embedding: list[float] | None = None,
) -> Chunk:
    return Chunk(
        document_id=document_id,
        position=position,
        kind="text",
        text=f"Chunk {position} of {document_id}.",
        path=[division.model_dump() for division in path],
        page=None,
        embedding=_vector(1.0) if embedding is None else embedding,
        embedding_model="text-embedding-3-small",
    )


def _save(session: Session, *rows: Document | Chunk) -> None:
    """Insert and commit, then forget the objects: the next reads come from the database."""
    session.add_all(rows)
    session.commit()
    session.expunge_all()


def _visible_to(session: Session, groups: list[str]) -> list[str]:
    statement = select(Document.id).where(Document.groups.overlap(groups)).order_by(Document.id)
    return list(session.scalars(statement))


@pytest.mark.integration
def test_a_document_and_its_chunks_read_back_unchanged(session: Session) -> None:
    internal = Document(
        id="breach-policy",
        title="Personal data breach response policy",
        short_title="Breach policy",
        source_url=None,  # internal documents have no public link
        regulations=["GDPR", "NIS2"],
        sha256="a" * 64,
        groups=["employees"],
    )
    table = "| Step | Deadline |\n| --- | --- |\n| Notify | 72 hours |"
    _save(
        session,
        internal,
        Chunk(
            document_id="breach-policy",
            position=1,
            kind="table",
            text=table,
            path=[Division(kind="heading", title="3 Deadlines").model_dump()],
            page=4,
            embedding=_vector(0.5, -0.25),  # exact in float32, the precision of a vector column
            embedding_model="text-embedding-3-small",
        ),
    )

    document = session.get(Document, "breach-policy")
    chunk = session.get(Chunk, ("breach-policy", 1))

    assert document is not None
    assert document.source_url is None
    assert document.regulations == ["GDPR", "NIS2"]
    assert document.groups == ["employees"]
    assert chunk is not None
    assert (chunk.kind, chunk.text, chunk.page) == ("table", table, 4)
    assert chunk.embedding == _vector(0.5, -0.25)
    assert chunk.embedding_model == "text-embedding-3-small"


@pytest.mark.integration
def test_a_document_inserts_its_chunks_and_loads_them_in_order(session: Session) -> None:
    document = _document()
    document.chunks = [_chunk(None, 1), _chunk(None, 0)]
    _save(session, document)

    loaded = session.get(Document, "gdpr")

    assert loaded is not None
    assert [(chunk.document_id, chunk.position) for chunk in loaded.chunks] == [
        ("gdpr", 0),
        ("gdpr", 1),
    ]


@pytest.mark.integration
def test_the_path_reads_back_into_divisions(session: Session) -> None:
    _save(session, _document(), _chunk())

    chunk = session.get(Chunk, ("gdpr", 0))

    assert chunk is not None
    assert TypeAdapter(tuple[Division, ...]).validate_python(chunk.path) == _ARTICLE_33


@pytest.mark.integration
def test_a_chunk_is_found_by_a_division_of_its_path(session: Session) -> None:
    article_34 = (*_ARTICLE_33[:2], Division(kind="article", number="34"))
    _save(session, _document(), _chunk(position=0), _chunk(position=1, path=article_34))

    statement = select(Chunk.position).where(
        Chunk.path.contains([{"kind": "article", "number": "34"}])  # JSONB containment: @>
    )

    assert list(session.scalars(statement)) == [1]


@pytest.mark.integration
def test_deleting_a_document_deletes_its_chunks_only(session: Session) -> None:
    _save(
        session,
        _document("gdpr"),
        _document("nis2"),
        _chunk("gdpr", 0),
        _chunk("gdpr", 1),
        _chunk("nis2", 0),
    )

    session.execute(delete(Document).where(Document.id == "gdpr"))
    session.commit()

    assert list(session.execute(select(Chunk.document_id, Chunk.position))) == [("nis2", 0)]


@pytest.mark.integration
def test_a_chunk_needs_an_existing_document(session: Session) -> None:
    session.add(_chunk("unknown"))

    with pytest.raises(IntegrityError, match="foreign key"):
        session.commit()


@pytest.mark.integration
def test_a_position_is_unique_within_a_document(session: Session) -> None:
    _save(session, _document("gdpr"), _document("nis2"), _chunk("gdpr", 0))
    _save(session, _chunk("nis2", 0))  # the same position in another document is fine

    session.add(_chunk("gdpr", 0))

    with pytest.raises(IntegrityError, match="duplicate key"):
        session.commit()


@pytest.mark.integration
def test_a_document_whose_groups_were_forgotten_is_not_inserted(session: Session) -> None:
    document = _document()
    del document.groups  # as if a connector forgot to set them
    session.add(document)

    with pytest.raises(IntegrityError, match="groups"):
        session.commit()


@pytest.mark.integration
def test_a_user_sees_the_documents_sharing_one_of_their_groups(session: Session) -> None:
    _save(session, _document("gdpr", ["public"]), _document("breach-policy", ["employees"]))

    assert _visible_to(session, ["public"]) == ["gdpr"]
    assert _visible_to(session, ["public", "employees"]) == ["breach-policy", "gdpr"]


@pytest.mark.integration
def test_a_document_without_groups_is_visible_to_nobody(session: Session) -> None:
    _save(session, _document("gdpr", []))

    assert _visible_to(session, ["public", "employees"]) == []
    assert session.scalar(select(func.count()).select_from(Document)) == 1


@pytest.mark.integration
def test_an_embedding_must_have_the_schema_dimensions(session: Session) -> None:
    _save(session, _document())
    session.add(_chunk(embedding=[0.1, 0.2, 0.3]))

    with pytest.raises(DataError, match=f"expected {EMBEDDING_DIMENSIONS} dimensions, not 3"):
        session.commit()
