from agentic_rag.document import Chunk
from agentic_rag.storage.base import IndexedDocument, RetrievedChunk, StoredChunk

_DOCUMENT = IndexedDocument(
    id="gdpr",
    title="GDPR",
    short_title="GDPR",
    source_url=None,
    regulations=(),
    groups=("public",),
    sha256="0",
    embedding_model="m",
)
_CHUNK = Chunk(kind="text", text="x", path=())


def test_a_chunk_is_named_by_its_document_and_position() -> None:
    assert StoredChunk(document=_DOCUMENT, position=12, chunk=_CHUNK).key == "gdpr#12"


def test_a_retrieved_chunk_is_a_stored_chunk_with_a_score() -> None:
    retrieved = RetrievedChunk(document=_DOCUMENT, position=3, chunk=_CHUNK, score=0.5)

    assert isinstance(retrieved, StoredChunk)
    assert retrieved.key == "gdpr#3"
