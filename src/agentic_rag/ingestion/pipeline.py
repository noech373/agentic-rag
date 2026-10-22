"""Ingestion pipeline: from the corpus manifest to the store, one document at a time.

parse (by format) > chunk > embed > store. A document is skipped when its file and the
embedding model are those it was stored with; a document gone from the manifest is deleted.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Literal

from agentic_rag.document import Block, Chunk
from agentic_rag.embeddings import Embedder
from agentic_rag.ingestion.chunking import chunk_blocks
from agentic_rag.ingestion.corpus import ChecksumMismatchError, DocumentFormat, SourceDocument
from agentic_rag.ingestion.eurlex import parse_eurlex
from agentic_rag.ingestion.markdown import parse_markdown
from agentic_rag.ingestion.pdf import parse_pdf
from agentic_rag.store import ChunkStore, IndexedDocument

type Parser = Callable[[bytes], list[Block]]
type Chunker = Callable[[Sequence[Block]], list[Chunk]]

_PUBLIC = ("public",)  # every document of the public corpus; internal ones will have their groups


def default_parsers(cache_dir: Path) -> dict[DocumentFormat, Parser]:
    return {
        "xhtml": parse_eurlex,
        "pdf": partial(parse_pdf, cache_dir=cache_dir),
        "markdown": parse_markdown,
    }


@dataclass(frozen=True)
class Outcome:
    document_id: str
    action: Literal["ingested", "unchanged", "deleted"]
    chunks: int | None = None  # how many were stored, for an ingested document


class Ingester:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        parsers: Mapping[DocumentFormat, Parser],
        chunker: Chunker = chunk_blocks,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._parsers = parsers
        self._chunker = chunker

    def run(
        self, documents: Sequence[SourceDocument], corpus_dir: Path, *, force: bool = False
    ) -> Iterator[Outcome]:
        """Bring the store in line with the manifest, reporting each document as it is done.

        Stops at the first error: the documents stored before it are skipped on the next run.
        `force` ingests unchanged documents too, e.g. after a change to the parsers or chunker.
        """
        self._store.create_schema()
        indexed = self._store.indexed_documents()
        for document in documents:
            stored = indexed.get(document.id)
            if not force and stored is not None and self._is_current(stored, document):
                yield Outcome(document.id, "unchanged")
                continue
            chunks = self.ingest(document, corpus_dir / document.filename)
            yield Outcome(document.id, "ingested", chunks)

        # All the stored documents come from this manifest for now: a second source will have
        # to delete only its own documents.
        gone = sorted(indexed.keys() - {document.id for document in documents})
        self._store.delete_documents(gone)
        for document_id in gone:
            yield Outcome(document_id, "deleted")

    def ingest(self, document: SourceDocument, path: Path) -> int:
        """Parse, chunk, embed and store one document; return its number of chunks."""
        # The manifest's checksum is what was downloaded: a file edited since then would make
        # the store, and the evaluation, depend on a local change.
        content = path.read_bytes()
        if sha256(content).hexdigest() != document.sha256:
            raise ChecksumMismatchError(f"{path} differs from the manifest: download it again")
        blocks = self._parsers[document.format](content)
        chunks = self._chunker(blocks)
        embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])
        self._store.replace_document(self._indexed(document), chunks, embeddings)
        return len(chunks)

    def _is_current(self, stored: IndexedDocument, document: SourceDocument) -> bool:
        return stored.sha256 == document.sha256 and stored.embedding_model == self._embedder.name

    def _indexed(self, document: SourceDocument) -> IndexedDocument:
        return IndexedDocument(
            id=document.id,
            title=document.title,
            short_title=document.short_title,
            source_url=str(document.source_url),
            regulations=document.regulations,
            groups=_PUBLIC,
            sha256=document.sha256,
            embedding_model=self._embedder.name,
        )
