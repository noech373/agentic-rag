import math
from collections.abc import Iterator, Sequence

import pytest
from pydantic import SecretStr, ValidationError
from sqlalchemy import Engine, Float, create_engine, func, insert, select, text
from sqlalchemy.exc import DataError, IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from agentic_rag.config import PostgresSettings
from agentic_rag.document import Chunk, Division
from agentic_rag.postgres import (
    BM25_INDEX,
    ChunkRow,
    DocumentRow,
    PostgresStore,
    database_url,
)
from agentic_rag.store import IndexedDocument

_ARTICLE_33 = (
    Division(kind="chapter", number="IV", title="Controller and processor"),
    Division(kind="section", number="2", title="Security of personal data"),
    Division(kind="article", number="33", title="Notification of a personal data breach"),
)


def test_the_database_url_comes_from_the_settings() -> None:
    settings = PostgresSettings(
        host="db", port=5433, user="rag", password=SecretStr("p@ss/word"), db="rag"
    )

    assert (
        database_url(settings).render_as_string(hide_password=False)
        == "postgresql+psycopg://rag:p%40ss%2Fword@db:5433/rag"
    )
    assert database_url(settings, "other").database == "other"


# --- On a real Postgres (skipped until `docker compose up -d` has been run) ---

_TEST_DATABASE = "agentic_rag_test"  # created, then dropped, on the server of the .env settings


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    """A fresh database with the schema, dropped after the tests."""
    try:
        settings = PostgresSettings()  # pyright: ignore[reportCallIssue]  (values come from env)
    except ValidationError:
        pytest.skip("POSTGRES_* settings not found: cp .env.example .env")
    # CREATE DATABASE cannot run inside a transaction, hence AUTOCOMMIT.
    admin = create_engine(
        database_url(settings), isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 3}
    )
    try:
        with admin.connect() as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS {_TEST_DATABASE} WITH (FORCE)"))
            connection.execute(text(f"CREATE DATABASE {_TEST_DATABASE}"))
    except OperationalError:
        admin.dispose()
        pytest.skip("Postgres not reachable: run `docker compose up -d`")

    engine = create_engine(database_url(settings, _TEST_DATABASE))
    PostgresStore(engine).create_schema()
    yield engine

    engine.dispose()
    with admin.connect() as connection:
        connection.execute(text(f"DROP DATABASE {_TEST_DATABASE} WITH (FORCE)"))
    admin.dispose()


@pytest.fixture
def store(engine: Engine) -> PostgresStore:
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE documents, chunks"))
    return PostgresStore(engine)


def _document(
    id_: str = "gdpr", *, sha256: str = "0" * 64, embedding_model: str = "model-a"
) -> IndexedDocument:
    return IndexedDocument(
        id=id_,
        title=f"Title of {id_}",
        short_title=id_.upper(),
        source_url=f"https://example.org/{id_}",
        regulations=("GDPR",),
        groups=("public",),
        sha256=sha256,
        embedding_model=embedding_model,
    )


def _chunk(text: str = "A chunk.", path: tuple[Division, ...] = _ARTICLE_33) -> Chunk:
    return Chunk(kind="text", text=text, path=path)


def _save(
    store: PostgresStore,
    document: IndexedDocument,
    *chunks: Chunk,
    embeddings: Sequence[Sequence[float]] | None = None,
) -> None:
    """Store the chunks, all with the same embedding unless given."""
    store.replace_document(document, chunks, embeddings or [[1.0, 0.0]] * len(chunks))


def _search(
    store: PostgresStore, query: Sequence[float], k: int = 10, embedding_model: str = "model-a"
) -> list[tuple[str, int]]:
    """(document id, position) of the results, best first."""
    results = store.vector_search(query, embedding_model=embedding_model, k=k)
    return [(result.document.id, result.position) for result in results]


@pytest.mark.integration
def test_create_schema_can_run_again(store: PostgresStore) -> None:
    _save(store, _document(), _chunk())

    store.create_schema()

    assert _search(store, [1.0, 0.0]) == [("gdpr", 0)]


@pytest.mark.integration
def test_drop_schema_then_create_schema_starts_empty(engine: Engine) -> None:
    store = PostgresStore(engine)
    _save(store, _document(), _chunk())

    store.drop_schema()
    store.create_schema()

    assert store.indexed_documents() == {}


@pytest.mark.integration
def test_a_document_reads_back_unchanged(store: PostgresStore) -> None:
    internal = IndexedDocument(
        id="breach-policy",
        title="Personal data breach response policy",
        short_title="Breach policy",
        source_url=None,  # internal documents have no public link
        regulations=("GDPR", "NIS2"),
        groups=("employees",),
        sha256="a" * 64,
        embedding_model="text-embedding-3-small@1536",
    )

    _save(store, internal, _chunk())

    assert store.indexed_documents() == {"breach-policy": internal}


@pytest.mark.integration
def test_chunks_read_back_unchanged_from_a_search(store: PostgresStore) -> None:
    text_chunk = _chunk("Article 33\n1. In the case of a personal data breach, ...")
    table = Chunk(
        kind="table",
        text="| Step | Deadline |\n| --- | --- |\n| Notify | 72 hours |",
        path=(Division(kind="heading", title="3 Deadlines"),),
        page=4,
    )
    _save(store, _document(), text_chunk, table, embeddings=[[1.0, 0.0], [0.0, 1.0]])

    results = store.vector_search([1.0, 0.1], embedding_model="model-a", k=10)

    assert [(r.document, r.position, r.chunk) for r in results] == [
        (_document(), 0, text_chunk),
        (_document(), 1, table),
    ]


@pytest.mark.integration
def test_a_document_without_chunks_is_stored(store: PostgresStore) -> None:
    _save(store, _document())

    assert list(store.indexed_documents()) == ["gdpr"]


@pytest.mark.integration
def test_replacing_a_document_replaces_all_its_chunks(store: PostgresStore) -> None:
    _save(store, _document(), _chunk("old 0"), _chunk("old 1"), _chunk("old 2"))

    _save(store, _document(sha256="b" * 64), _chunk("new 0"))

    results = store.vector_search([1.0, 0.0], embedding_model="model-a", k=10)
    assert [result.chunk.text for result in results] == ["new 0"]
    assert store.indexed_documents()["gdpr"].sha256 == "b" * 64


@pytest.mark.integration
def test_replacing_a_document_leaves_the_others_untouched(store: PostgresStore) -> None:
    _save(store, _document("gdpr"), _chunk())
    _save(store, _document("nis2"), _chunk(), _chunk())

    _save(store, _document("gdpr", sha256="b" * 64), _chunk())

    assert sorted(_search(store, [1.0, 0.0])) == [("gdpr", 0), ("nis2", 0), ("nis2", 1)]


@pytest.mark.integration
def test_a_failed_replacement_keeps_the_previous_version(store: PostgresStore) -> None:
    _save(store, _document(), _chunk("old"))

    with pytest.raises(DataError, match="NaN not allowed"):  # rejected after the old rows' delete
        _save(store, _document(sha256="b" * 64), _chunk("new"), embeddings=[[math.nan, 0.0]])

    assert store.indexed_documents()["gdpr"].sha256 == "0" * 64
    assert _search(store, [1.0, 0.0]) == [("gdpr", 0)]


@pytest.mark.integration
def test_each_chunk_needs_one_embedding(store: PostgresStore) -> None:
    with pytest.raises(ValueError, match="zip"):
        _save(store, _document(), _chunk(), _chunk(), embeddings=[[1.0, 0.0]])

    assert store.indexed_documents() == {}


@pytest.mark.integration
def test_deleting_documents_deletes_their_chunks_only(store: PostgresStore) -> None:
    for id_ in ("gdpr", "nis2", "dora"):
        _save(store, _document(id_), _chunk(), _chunk())

    store.delete_documents(["gdpr", "dora", "unknown"])

    assert list(store.indexed_documents()) == ["nis2"]
    assert sorted(_search(store, [1.0, 0.0])) == [("nis2", 0), ("nis2", 1)]


@pytest.mark.integration
def test_vector_search_ranks_by_cosine_similarity(store: PostgresStore) -> None:
    _save(
        store,
        _document(),
        _chunk(),
        _chunk(),
        _chunk(),
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )

    results = store.vector_search([1.0, 0.0], embedding_model="model-a", k=10)

    assert [result.position for result in results] == [0, 2, 1]
    assert [result.score for result in results] == pytest.approx([1.0, math.sqrt(0.5), 0.0])


@pytest.mark.integration
def test_vector_search_returns_at_most_k_chunks(store: PostgresStore) -> None:
    _save(store, _document(), *[_chunk() for _ in range(5)])

    assert len(_search(store, [1.0, 0.0], k=3)) == 3


@pytest.mark.integration
def test_vector_search_only_compares_vectors_of_the_query_model(store: PostgresStore) -> None:
    """Models of different dimensions share the column: comparing them would raise an error."""
    _save(store, _document("gdpr", embedding_model="model-a"), _chunk(), embeddings=[[1.0, 0.0]])
    _save(
        store, _document("nis2", embedding_model="model-b"), _chunk(), embeddings=[[1.0, 0.0, 0.0]]
    )

    assert _search(store, [1.0, 0.0, 0.0], embedding_model="model-b") == [("nis2", 0)]
    assert _search(store, [1.0, 0.0], embedding_model="model-a") == [("gdpr", 0)]


# --- Schema guarantees, beyond what the store writes ---


def _insert_document(engine: Engine, **values: object) -> None:
    with engine.begin() as connection:
        connection.execute(insert(DocumentRow).values(**values))


_ROW = {
    "id": "gdpr",
    "title": "GDPR",
    "short_title": "GDPR",
    "regulations": ["GDPR"],
    "sha256": "0" * 64,
    "embedding_model": "model-a",
}


@pytest.mark.integration
def test_a_document_whose_groups_were_forgotten_is_not_inserted(
    store: PostgresStore, engine: Engine
) -> None:
    with pytest.raises(IntegrityError, match="groups"):
        _insert_document(engine, **_ROW)


@pytest.mark.integration
def test_a_chunk_needs_an_existing_document(store: PostgresStore, engine: Engine) -> None:
    with pytest.raises(IntegrityError, match="foreign key"), engine.begin() as connection:
        connection.execute(
            insert(ChunkRow).values(
                document_id="unknown",
                position=0,
                kind="text",
                text="A chunk.",
                path=[],
                embedding=[1.0, 0.0],
            ),
        )


def _visible_to(engine: Engine, groups: list[str]) -> list[str]:
    statement = (
        select(DocumentRow.id).where(DocumentRow.groups.overlap(groups)).order_by(DocumentRow.id)
    )
    with Session(engine) as session:
        return list(session.scalars(statement))


@pytest.mark.integration
def test_a_user_sees_the_documents_sharing_one_of_their_groups(
    store: PostgresStore, engine: Engine
) -> None:
    _insert_document(engine, **_ROW, groups=["public"])
    _insert_document(engine, **{**_ROW, "id": "breach-policy"}, groups=["employees"])

    assert _visible_to(engine, ["public"]) == ["gdpr"]
    assert _visible_to(engine, ["public", "employees"]) == ["breach-policy", "gdpr"]


@pytest.mark.integration
def test_a_document_without_groups_is_visible_to_nobody(
    store: PostgresStore, engine: Engine
) -> None:
    _insert_document(engine, **_ROW, groups=[])

    assert _visible_to(engine, ["public", "employees"]) == []
    assert store.indexed_documents().keys() == {"gdpr"}


@pytest.mark.integration
def test_a_chunk_is_found_by_a_division_of_its_path(store: PostgresStore, engine: Engine) -> None:
    article_34 = (*_ARTICLE_33[:2], Division(kind="article", number="34"))
    _save(store, _document(), _chunk(), _chunk(path=article_34))

    statement = select(ChunkRow.position).where(
        ChunkRow.path.contains([{"kind": "article", "number": "34"}])  # JSONB containment: @>
    )

    with Session(engine) as session:
        assert list(session.scalars(statement)) == [1]


# --- BM25 (pg_textsearch): checked here, before a retriever uses it ---


def _bm25(engine: Engine, query: str) -> list[tuple[int, float]]:
    """(position, score) of every chunk, best first: scores are negative, 0 with no common term."""
    score = ChunkRow.text.op("<@>", return_type=Float())(func.to_bm25query(query, BM25_INDEX))
    statement = select(ChunkRow.position, score).order_by(score, ChunkRow.position)
    with Session(engine) as session:
        return [(position, value) for position, value in session.execute(statement)]


def _ranked(engine: Engine, query: str) -> list[int]:
    """The positions of the chunks sharing a term with the query, best first."""
    return [position for position, score in _bm25(engine, query) if score < 0]


def _texts(store: PostgresStore, *texts: str) -> None:
    _save(store, _document(), *(_chunk(text) for text in texts))


@pytest.mark.integration
def test_bm25_matches_the_stems_of_the_query(store: PostgresStore, engine: Engine) -> None:
    _texts(
        store,
        "The data subject has the right to erasure.",
        "The controller shall notify the authority of a personal data breach.",
    )

    assert _ranked(engine, "notified breaches") == [1]


@pytest.mark.integration
def test_bm25_ignores_case_and_stop_words(store: PostgresStore, engine: Engine) -> None:
    _texts(store, "The controller shall notify the authority of a personal data breach.")

    assert _bm25(engine, "THE Breach") == _bm25(engine, "breach")


@pytest.mark.integration
def test_bm25_weighs_a_rare_term_more_than_a_common_one(
    store: PostgresStore, engine: Engine
) -> None:
    _texts(
        store,
        "personal data processing",
        "personal data controller",
        "personal data security",  # "personal" is in 3 chunks out of 4
        "breach data security",  # "breach" in only 1
    )

    assert _ranked(engine, "personal breach")[0] == 3


@pytest.mark.integration
def test_bm25_sees_inserted_and_deleted_chunks_at_once(
    store: PostgresStore, engine: Engine
) -> None:
    _texts(store, "The controller shall notify the authority of a personal data breach.")
    assert _ranked(engine, "breach") == [0]

    store.delete_documents(["gdpr"])

    assert _ranked(engine, "breach") == []


@pytest.mark.integration
def test_the_short_bm25_syntax_fails_with_a_bound_parameter(
    store: PostgresStore, engine: Engine
) -> None:
    """Why queries name the index: `text <@> :query` cannot find it (pg_textsearch 1.5.1)."""
    _texts(store, "The controller shall notify the authority of a personal data breach.")
    score = ChunkRow.text.op("<@>", return_type=Float())("breach")

    with pytest.raises(ProgrammingError, match="no BM25 index found"), Session(engine) as session:
        session.execute(select(ChunkRow.position).order_by(score))
