"""Public corpus: the manifest of source documents and their verified download."""

import hashlib
import tomllib
from collections import Counter
from pathlib import Path
from typing import Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

type DocumentFormat = Literal["xhtml", "pdf", "markdown"]

_EXTENSIONS: dict[DocumentFormat, str] = {"xhtml": ".xhtml", "pdf": ".pdf", "markdown": ".md"}


class SourceDocument(BaseModel):
    """One entry of the manifest: where to download a document and how to cite it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")  # a slug: it becomes the file name
    title: str
    short_title: str
    publisher: str
    doc_type: Literal["regulation", "directive", "guidelines", "report", "guide"]
    regulations: tuple[str, ...]  # a tuple, because frozen=True does not freeze a list
    format: DocumentFormat
    url: HttpUrl
    source_url: HttpUrl
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def filename(self) -> str:
        return self.id + _EXTENSIONS[self.format]


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: tuple[SourceDocument, ...]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        counts = Counter(document.id for document in self.documents)
        duplicates = sorted(id_ for id_, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate document ids: {duplicates}")
        return self


def load_manifest(path: Path) -> Manifest:
    with path.open("rb") as f:
        return Manifest.model_validate(tomllib.load(f))


class ChecksumMismatchError(Exception):
    pass


def sha256_of(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(document: SourceDocument, target_dir: Path, client: httpx.Client) -> bool:
    """Download `document` into `target_dir`, unless a verified copy is already there.

    Returns True if the file was downloaded, False if the local copy was already valid.
    Raises ChecksumMismatchError if the downloaded bytes differ from the manifest; nothing is
    kept in that case, nor when the download fails halfway.
    """
    path = target_dir / document.filename
    if path.exists() and sha256_of(path) == document.sha256:
        return False

    partial = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    try:
        with client.stream("GET", str(document.url)) as response:
            response.raise_for_status()
            with partial.open("wb") as f:
                for chunk in response.iter_bytes():
                    digest.update(chunk)
                    f.write(chunk)
        if digest.hexdigest() != document.sha256:
            raise ChecksumMismatchError(
                f"expected sha256 {document.sha256}, got {digest.hexdigest()}"
            )
        # Atomic rename: the final name only ever holds a complete, verified file, even if the
        # process is killed halfway through the download.
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)  # a failed or rejected download leaves nothing behind
    return True
