import hashlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from agentic_rag.cli import app
from agentic_rag.ingestion.corpus import (
    ChecksumMismatchError,
    Manifest,
    SourceDocument,
    download,
    load_manifest,
)

_MANIFEST = Path(__file__).parents[2] / "corpus" / "manifest.toml"
_CONTENT = b"Article 33: notify the breach within 72 hours."


def _document(**overrides: str) -> SourceDocument:
    """A valid document whose checksum matches _CONTENT. Override a field to break it."""
    fields = {
        "id": "gdpr",
        "title": "General Data Protection Regulation",
        "short_title": "GDPR",
        "publisher": "European Union",
        "doc_type": "regulation",
        "regulations": ["GDPR"],
        "format": "xhtml",
        "url": "https://example.org/gdpr",
        "source_url": "https://example.org/gdpr",
        "sha256": hashlib.sha256(_CONTENT).hexdigest(),
    } | overrides
    # model_validate, not SourceDocument(...): pyright strict rejects a str for an HttpUrl field.
    return SourceDocument.model_validate(fields)


def _client(status_code: int = 200) -> tuple[httpx.Client, list[httpx.Request]]:
    """An HTTP client that never touches the network and records the requests it receives."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, content=_CONTENT)

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


# --- Manifest ---


def test_committed_manifest_is_valid() -> None:
    documents = load_manifest(_MANIFEST).documents

    assert len(documents) > 0


def test_committed_manifest_file_names_are_unique() -> None:
    documents = load_manifest(_MANIFEST).documents

    assert len({document.filename for document in documents}) == len(documents)


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match=r"duplicate document ids: \['gdpr'\]"):
        Manifest(documents=(_document(), _document(), _document(id="nis2")))


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="sha265"):
        _document(sha265="typo")


@pytest.mark.parametrize("bad_id", ["GDPR", "gdpr_2016", "-gdpr", "gdpr--act", "../gdpr"])
def test_id_must_be_a_slug(bad_id: str) -> None:
    # The id becomes a file name: no path separators, no surprises.
    with pytest.raises(ValidationError, match="id"):
        _document(id=bad_id)


def test_filename_depends_on_the_format() -> None:
    assert _document(format="pdf").filename == "gdpr.pdf"
    assert _document(format="markdown").filename == "gdpr.md"


# --- Download ---


def test_downloads_and_verifies(tmp_path: Path) -> None:
    client, requests = _client()

    assert download(_document(), tmp_path, client) is True

    assert (tmp_path / "gdpr.xhtml").read_bytes() == _CONTENT
    assert [str(request.url) for request in requests] == ["https://example.org/gdpr"]


def test_skips_a_verified_local_copy(tmp_path: Path) -> None:
    client, requests = _client()
    download(_document(), tmp_path, client)

    assert download(_document(), tmp_path, client) is False
    assert len(requests) == 1  # no second request


def test_replaces_a_corrupted_local_copy(tmp_path: Path) -> None:
    (tmp_path / "gdpr.xhtml").write_bytes(b"truncated")
    client, _ = _client()

    assert download(_document(), tmp_path, client) is True
    assert (tmp_path / "gdpr.xhtml").read_bytes() == _CONTENT


def test_checksum_mismatch_keeps_nothing(tmp_path: Path) -> None:
    client, _ = _client()

    with pytest.raises(ChecksumMismatchError):
        download(_document(sha256="0" * 64), tmp_path, client)

    assert list(tmp_path.iterdir()) == []


def test_http_error_keeps_nothing(tmp_path: Path) -> None:
    client, _ = _client(status_code=404)

    with pytest.raises(httpx.HTTPStatusError):
        download(_document(), tmp_path, client)

    assert list(tmp_path.iterdir()) == []


# --- CLI ---


_MANIFEST_ENTRY = """
[[documents]]
id = "{id}"
title = "Test document"
short_title = "TEST"
publisher = "European Union"
doc_type = "regulation"
regulations = ["GDPR"]
format = "xhtml"
url = "https://example.org/{id}"
source_url = "https://example.org/{id}"
sha256 = "{sha256}"
"""


def test_cli_downloads_the_others_and_fails_when_one_document_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every client the CLI creates answers _CONTENT without touching the network.
    real_client = httpx.Client

    def offline_client(**kwargs: Any) -> httpx.Client:
        handler = httpx.MockTransport(lambda _: httpx.Response(200, content=_CONTENT))
        return real_client(transport=handler, **kwargs)

    monkeypatch.setattr(httpx, "Client", offline_client)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _MANIFEST_ENTRY.format(id="gdpr", sha256=hashlib.sha256(_CONTENT).hexdigest())
        + _MANIFEST_ENTRY.format(id="nis2", sha256="0" * 64)  # cannot match: rejected
    )
    target_dir = tmp_path / "corpus"

    result = CliRunner().invoke(
        app, ["download", "--manifest", str(manifest), "--target-dir", str(target_dir)]
    )

    assert result.exit_code == 1
    assert "downloaded  gdpr" in result.stdout
    assert "FAILED      nis2" in result.stderr
    assert "1 of 2 documents failed" in result.stderr
    assert [path.name for path in target_dir.iterdir()] == ["gdpr.xhtml"]
