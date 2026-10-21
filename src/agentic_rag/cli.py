"""Command-line interface: `uv run agentic-rag --help`."""

import time
from pathlib import Path
from typing import Annotated

import httpx
import typer

from agentic_rag.ingestion.corpus import ChecksumMismatchError, download, load_manifest

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Agentic RAG over EU regulatory compliance documents."""
    # With a single command, Typer would make it the root command and `agentic-rag download`
    # would fail. This callback keeps commands as subcommands; its docstring is the --help text.


@app.command("download")
def download_corpus(
    manifest: Annotated[Path, typer.Option(help="Corpus manifest.")] = Path("corpus/manifest.toml"),
    target_dir: Annotated[Path, typer.Option(help="Download directory.")] = Path("data/corpus"),
) -> None:
    """Download the public corpus and verify each file against its SHA-256."""

    documents = load_manifest(manifest).documents
    target_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for document in documents:
            try:
                downloaded = download(document, target_dir, client)
            except (httpx.HTTPError, ChecksumMismatchError) as exc:
                # One bad source must not prevent fetching the others.
                failures += 1
                typer.echo(f"FAILED      {document.id}: {exc}", err=True)
                continue
            typer.echo(f"{'downloaded' if downloaded else 'up to date':<11} {document.id}")

    if failures:
        typer.echo(f"{failures} of {len(documents)} documents failed", err=True)
        raise typer.Exit(code=1)


@app.command("convert")
def convert_pdfs(
    manifest: Annotated[Path, typer.Option(help="Corpus manifest.")] = Path("corpus/manifest.toml"),
    corpus_dir: Annotated[Path, typer.Option(help="Downloaded corpus.")] = Path("data/corpus"),
    cache_dir: Annotated[Path, typer.Option(help="Docling cache.")] = Path("data/cache"),
) -> None:
    """Run Docling on the PDFs of the corpus and cache its output (minutes per document)."""
    # Imported here: Docling takes seconds to import, the other commands do not need it.
    from agentic_rag.ingestion.pdf import cache_path, load_or_convert

    for document in load_manifest(manifest).documents:
        if document.format != "pdf":
            continue
        content = (corpus_dir / document.filename).read_bytes()
        if cache_path(content, cache_dir).exists():
            typer.echo(f"{'up to date':<11} {document.id}")
            continue
        start = time.perf_counter()
        load_or_convert(content, cache_dir)
        typer.echo(f"{'converted':<11} {document.id} in {time.perf_counter() - start:.0f} s")
