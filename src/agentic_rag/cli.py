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
    from agentic_rag.ingestion.parsers.pdf import cache_path, load_or_convert

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


@app.command("ingest")
def ingest_corpus(
    manifest: Annotated[Path, typer.Option(help="Corpus manifest.")] = Path("corpus/manifest.toml"),
    corpus_dir: Annotated[Path, typer.Option(help="Downloaded corpus.")] = Path("data/corpus"),
    cache_dir: Annotated[Path, typer.Option(help="Docling cache.")] = Path("data/cache"),
    force: Annotated[
        bool, typer.Option(help="Also re-ingest unchanged documents (after a parser change).")
    ] = False,
) -> None:
    """Parse, chunk and embed the corpus into the store; unchanged documents are skipped."""
    from agentic_rag.config import get_settings
    from agentic_rag.factory import make_embedder, make_store
    from agentic_rag.ingestion.pipeline import Ingester, default_parsers  # imports Docling

    settings = get_settings()
    ingester = Ingester(make_store(settings), make_embedder(settings), default_parsers(cache_dir))
    documents = load_manifest(manifest).documents
    start = time.perf_counter()
    for outcome in ingester.run(documents, corpus_dir, force=force):
        details = f" ({outcome.chunks} chunks)" if outcome.chunks is not None else ""
        typer.echo(f"{outcome.action:<11} {outcome.document_id}{details}")
    typer.echo(f"done in {time.perf_counter() - start:.0f} s")


@app.command("ask")
def ask(
    question: Annotated[str, typer.Argument(help="A question on EU compliance.")],
    k: Annotated[int, typer.Option(min=1, help="Number of sources given to the LLM.")] = 8,
    retrieved: Annotated[
        bool, typer.Option(help="List every retrieved source with its score, cited or not.")
    ] = False,
) -> None:
    """Answer a question from the ingested corpus, citing the sources."""
    from agentic_rag.config import get_settings
    from agentic_rag.factory import make_rag
    from agentic_rag.generation.prompt import citation

    if not question.strip():
        raise typer.BadParameter("the question is empty")
    answer = make_rag(get_settings(), k=k).ask(question)
    typer.echo(answer.text)

    cited = answer.cited
    sources = dict(enumerate(answer.sources, start=1)) if retrieved else cited
    if sources:
        typer.echo()
    for number, source in sources.items():
        url = f" {source.document.source_url}" if source.document.source_url else ""
        score = f"{source.score:.3f} " if retrieved else ""
        mark = "" if not retrieved or number in cited else " (not cited)"
        typer.echo(f"[{number}] {score}{citation(source)}{url}{mark}")
