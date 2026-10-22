from collections.abc import Collection, Sequence
from hashlib import sha256
from pathlib import Path
from typing import get_args

import pytest

from agentic_rag.document import Block, Chunk, Division
from agentic_rag.embeddings import Embedder
from agentic_rag.ingestion.corpus import ChecksumMismatchError, DocumentFormat, SourceDocument
from agentic_rag.ingestion.pipeline import Ingester, Outcome, Parser, default_parsers
from agentic_rag.store import ChunkStore, IndexedDocument


class _FakeStore(ChunkStore):
    def __init__(self) -> None:
        self.schema_created = False
        self.documents: dict[str, IndexedDocument] = {}
        self.chunks: dict[str, list[tuple[Chunk, list[float]]]] = {}

    def create_schema(self) -> None:
        self.schema_created = True

    def indexed_documents(self) -> dict[str, IndexedDocument]:
        return dict(self.documents)

    def replace_document(
        self,
        document: IndexedDocument,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.documents[document.id] = document
        self.chunks[document.id] = [
            (chunk, list(embedding)) for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

    def delete_documents(self, ids: Collection[str]) -> None:
        for id_ in ids:
            del self.documents[id_], self.chunks[id_]


class _FakeEmbedder(Embedder):
    """The vector of a text is [its length, 1]."""

    def __init__(self, name: str = "fake@2") -> None:
        self._name = name
        self.calls: list[list[str]] = []

    @property
    def name(self) -> str:
        return self._name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _lines(content: bytes) -> list[Block]:
    """A parser: one block per line."""
    return [Block(kind="text", text=line, path=()) for line in content.decode().splitlines()]


_PARSERS: dict[DocumentFormat, Parser] = {"xhtml": _lines, "pdf": _lines, "markdown": _lines}


def _one_chunk_per_block(blocks: Sequence[Block]) -> list[Chunk]:
    return [Chunk(kind=block.kind, text=block.text, path=block.path) for block in blocks]


def _ingester(
    store: _FakeStore, embedder: _FakeEmbedder, parsers: dict[DocumentFormat, Parser] = _PARSERS
) -> Ingester:
    return Ingester(store, embedder, parsers, chunker=_one_chunk_per_block)


def _source(
    corpus: Path, id_: str, content: str, format_: DocumentFormat = "markdown"
) -> SourceDocument:
    """A manifest entry, and its file written in the corpus directory."""
    document = SourceDocument(
        id=id_,
        title=f"Title of {id_}",
        short_title=id_.upper(),
        publisher="Publisher",
        doc_type="guide",
        regulations=("GDPR", "NIS2"),
        format=format_,
        url=f"https://example.org/{id_}/download",  # pyright: ignore[reportArgumentType]
        source_url=f"https://example.org/{id_}",  # pyright: ignore[reportArgumentType]
        sha256=sha256(content.encode()).hexdigest(),
    )
    (corpus / document.filename).write_text(content)
    return document


def _run(
    ingester: Ingester, documents: Sequence[SourceDocument], corpus: Path, *, force: bool = False
) -> list[tuple[str, str, int | None]]:
    outcomes = ingester.run(documents, corpus, force=force)
    return [(outcome.document_id, outcome.action, outcome.chunks) for outcome in outcomes]


@pytest.fixture
def store() -> _FakeStore:
    return _FakeStore()


@pytest.fixture
def embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


def test_every_document_of_the_manifest_is_ingested(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    documents = [_source(tmp_path, "gdpr", "line 1\nline 22"), _source(tmp_path, "nis2", "one")]

    outcomes = _run(_ingester(store, embedder), documents, tmp_path)

    assert outcomes == [("gdpr", "ingested", 2), ("nis2", "ingested", 1)]
    assert store.schema_created
    assert store.chunks["gdpr"] == [
        (Chunk(kind="text", text="line 1", path=()), [6.0, 1.0]),
        (Chunk(kind="text", text="line 22", path=()), [7.0, 1.0]),
    ]


def test_a_stored_document_keeps_its_manifest_entry(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    document = _source(tmp_path, "gdpr", "text")

    _run(_ingester(store, embedder), [document], tmp_path)

    assert store.documents["gdpr"] == IndexedDocument(
        id="gdpr",
        title="Title of gdpr",
        short_title="GDPR",
        source_url="https://example.org/gdpr",
        regulations=("GDPR", "NIS2"),
        groups=("public",),
        sha256=document.sha256,
        embedding_model="fake@2",
    )


def test_the_chunks_of_a_document_are_embedded_in_one_call(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    documents = [_source(tmp_path, "gdpr", "a\nb\nc"), _source(tmp_path, "nis2", "d")]

    _run(_ingester(store, embedder), documents, tmp_path)

    assert embedder.calls == [["a", "b", "c"], ["d"]]


def test_the_parser_is_chosen_by_the_format_of_the_document(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    def tagged(format_: str) -> Parser:
        return lambda content: [Block(kind="text", text=f"{format_}: {content.decode()}", path=())]

    parsers: dict[DocumentFormat, Parser] = {f: tagged(f) for f in ("xhtml", "pdf", "markdown")}
    documents = [
        _source(tmp_path, "gdpr", "act", "xhtml"),
        _source(tmp_path, "edpb", "guidelines", "pdf"),
        _source(tmp_path, "cnil", "sheet", "markdown"),
    ]

    _run(_ingester(store, embedder, parsers), documents, tmp_path)

    assert embedder.calls == [["xhtml: act"], ["pdf: guidelines"], ["markdown: sheet"]]


def test_every_format_of_the_manifest_has_a_default_parser(tmp_path: Path) -> None:
    assert default_parsers(tmp_path).keys() == set(get_args(DocumentFormat.__value__))


def test_the_default_chunker_merges_the_blocks_of_an_article(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    article = (Division(kind="article", number="33"),)

    def paragraphs(content: bytes) -> list[Block]:
        return [Block(kind="text", text=line, path=article) for line in content.decode().split()]

    ingester = Ingester(store, embedder, {"markdown": paragraphs})

    assert _run(ingester, [_source(tmp_path, "gdpr", "1. 2. 3.")], tmp_path) == [
        ("gdpr", "ingested", 1)
    ]


def test_an_unchanged_document_is_not_ingested_again(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    documents = [_source(tmp_path, "gdpr", "text"), _source(tmp_path, "nis2", "text")]
    ingester = _ingester(store, embedder)
    _run(ingester, documents, tmp_path)
    embedder.calls.clear()

    outcomes = _run(ingester, documents, tmp_path)

    assert outcomes == [("gdpr", "unchanged", None), ("nis2", "unchanged", None)]
    assert embedder.calls == []


def test_a_document_whose_file_changed_is_ingested_again(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    ingester = _ingester(store, embedder)
    _run(ingester, [_source(tmp_path, "gdpr", "old"), _source(tmp_path, "nis2", "text")], tmp_path)

    outcomes = _run(
        ingester,
        [_source(tmp_path, "gdpr", "new\nversion"), _source(tmp_path, "nis2", "text")],
        tmp_path,
    )

    assert outcomes == [("gdpr", "ingested", 2), ("nis2", "unchanged", None)]
    assert [chunk.text for chunk, _ in store.chunks["gdpr"]] == ["new", "version"]


def test_another_embedding_model_ingests_every_document_again(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    documents = [_source(tmp_path, "gdpr", "text"), _source(tmp_path, "nis2", "text")]
    _run(_ingester(store, embedder), documents, tmp_path)

    outcomes = _run(_ingester(store, _FakeEmbedder("other@3")), documents, tmp_path)

    assert outcomes == [("gdpr", "ingested", 1), ("nis2", "ingested", 1)]
    assert {document.embedding_model for document in store.documents.values()} == {"other@3"}


def test_force_ingests_unchanged_documents_too(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    documents = [_source(tmp_path, "gdpr", "text")]
    ingester = _ingester(store, embedder)
    _run(ingester, documents, tmp_path)

    assert _run(ingester, documents, tmp_path, force=True) == [("gdpr", "ingested", 1)]


def test_a_document_gone_from_the_manifest_is_deleted(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    gdpr, nis2, dora = (_source(tmp_path, id_, "text") for id_ in ("gdpr", "nis2", "dora"))
    ingester = _ingester(store, embedder)
    _run(ingester, [gdpr, nis2, dora], tmp_path)

    outcomes = _run(ingester, [nis2], tmp_path)

    assert outcomes == [
        ("nis2", "unchanged", None),
        ("dora", "deleted", None),
        ("gdpr", "deleted", None),
    ]
    assert list(store.documents) == ["nis2"]


def test_a_file_that_differs_from_the_manifest_is_refused(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    document = _source(tmp_path, "gdpr", "downloaded")
    (tmp_path / document.filename).write_text("edited since")

    with pytest.raises(ChecksumMismatchError, match="differs from the manifest"):
        _run(_ingester(store, embedder), [document], tmp_path)

    assert store.documents == {}
    assert embedder.calls == []


def test_a_failure_stops_the_run_and_the_next_run_resumes_after_it(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    def fails_on_bad(content: bytes) -> list[Block]:
        if content == b"bad":
            raise ValueError("unexpected structure")
        return _lines(content)

    parsers: dict[DocumentFormat, Parser] = {"markdown": fails_on_bad}
    gdpr, nis2 = _source(tmp_path, "gdpr", "good"), _source(tmp_path, "nis2", "bad")
    with pytest.raises(ValueError, match="unexpected structure"):
        _run(_ingester(store, embedder, parsers), [gdpr, nis2], tmp_path)
    assert list(store.documents) == ["gdpr"]

    fixed = _source(tmp_path, "nis2", "fixed")
    outcomes = _run(_ingester(store, embedder, parsers), [gdpr, fixed], tmp_path)

    assert outcomes == [("gdpr", "unchanged", None), ("nis2", "ingested", 1)]


def test_outcomes_come_as_each_document_is_stored(
    store: _FakeStore, embedder: _FakeEmbedder, tmp_path: Path
) -> None:
    documents = [_source(tmp_path, "gdpr", "text"), _source(tmp_path, "nis2", "text")]
    outcomes = _ingester(store, embedder).run(documents, tmp_path)

    assert next(outcomes) == Outcome("gdpr", "ingested", 1)
    assert list(store.documents) == ["gdpr"]  # nis2 is not processed yet
