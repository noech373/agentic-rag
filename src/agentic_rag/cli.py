"""Command-line interface: `uv run agentic-rag --help`."""

import time
from pathlib import Path
from typing import Annotated

import httpx
import typer
from pydantic import ValidationError

from agentic_rag.evaluation.record import RunRecord, load_run
from agentic_rag.ingestion.corpus import ChecksumMismatchError, download, load_manifest
from agentic_rag.pipeline_config import PipelineConfig, load_pipeline

app = typer.Typer(no_args_is_help=True)

# Shared by `ask` and `eval`, so both build the same RAG from the same file.
PipelineOption = Annotated[
    Path | None,
    typer.Option(
        "--pipeline",
        exists=True,
        dir_okay=False,
        help="Pipeline TOML file (pipelines/<name>.toml). Default: the baseline.",
    ),
]


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
    pipeline: PipelineOption = None,
) -> None:
    """Answer a question from the ingested corpus, citing the sources."""
    from agentic_rag.citation import citation
    from agentic_rag.config import get_settings
    from agentic_rag.factory import make_rag

    if not question.strip():
        raise typer.BadParameter("the question is empty")
    config = _load_pipeline(pipeline)  # before the settings: a bad file fails fast
    answer = make_rag(get_settings(), config, k=k).ask(question)
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


@app.command("eval")
def evaluate(
    pipeline: PipelineOption = None,
    golden: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Golden set.")] = Path(
        "eval/golden.toml"
    ),
    judgments: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Rejected chunks, for judged@k.")
    ] = Path("eval/judgments.toml"),
) -> None:
    """Score the retrieval on the tuning questions and save the run; see also `compare`."""
    from agentic_rag.config import get_settings
    from agentic_rag.evaluation import comparison, record
    from agentic_rag.factory import make_retriever

    config = _load_pipeline(pipeline)
    settings = get_settings()
    name = pipeline.stem if pipeline is not None else "baseline"

    run = record.record_run(
        golden, judgments, lambda p: make_retriever(settings, p), config, Path()
    )
    path = record.RESULTS_DIR / f"{name}.json"
    record.save_run(run, path)
    typer.echo(f"saved {path}\n")
    typer.echo(comparison.summary_table(run, name))


RunArgument = Annotated[Path, typer.Argument(exists=True, dir_okay=False)]


@app.command("compare")
def compare(baseline: RunArgument, candidate: RunArgument) -> None:
    """Compare two saved runs (eval/results/<name>.json): the candidate against the baseline."""
    from agentic_rag.evaluation import comparison

    before = _load_run(baseline, "BASELINE")
    after = _load_run(candidate, "CANDIDATE")
    try:
        warnings = comparison.check_comparable(before, after)
    except comparison.IncomparableRunsError as exc:
        typer.echo(f"not compared: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(comparison.comparison_table(before, after, baseline.stem, candidate.stem))
    for warning in warnings:
        typer.echo(f"\nwarning: {warning}", err=True)


@app.command("pool")
def pool_runs(
    runs: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False, help="Saved runs.")],
    golden: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Golden set.")] = Path(
        "eval/golden.toml"
    ),
    judgments: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Rejected chunks.")
    ] = Path("eval/judgments.toml"),
    limit: Annotated[int | None, typer.Option(help="Judge at most this many chunks.")] = None,
) -> None:
    """Pre-judge the unjudged top chunks of the runs with the LLM, then write the review to fill in.

    Run again to resume after a stop.
    """
    from agentic_rag.config import get_settings
    from agentic_rag.evaluation import pooling
    from agentic_rag.evaluation.golden import load_golden_set
    from agentic_rag.evaluation.judgments import load_judgments
    from agentic_rag.evaluation.record import file_sha256
    from agentic_rag.evaluation.runner import tuning_questions
    from agentic_rag.factory import make_relevance_judge, make_store

    records = [_load_run(path, "RUNS") for path in runs]
    golden_sha256 = file_sha256(golden)
    stale = [
        path.name
        for path, run in zip(runs, records, strict=True)
        if run.golden_sha256 != golden_sha256
    ]
    if stale:
        raise typer.BadParameter(f"on another golden set: {', '.join(stale)}", param_hint="RUNS")
    llm_path = pooling.POOLING_DIR / "llm.jsonl"
    review_path = pooling.POOLING_DIR / "review.toml"
    if review_path.exists():  # before spending anything on the LLM
        typer.echo(f"{review_path} exists: it may hold a review, rename or delete it", err=True)
        raise typer.Exit(code=1)

    settings = get_settings()
    questions = tuning_questions(load_golden_set(golden))
    chunks = {chunk.key: chunk for chunk in make_store(settings).all_chunks()}
    done = pooling.load_pool_judgments(llm_path)
    pending = pooling.not_yet_judged(
        pooling.chunks_to_judge(questions, records, load_judgments(judgments), chunks), done
    )
    typer.echo(f"{len(pending)} pairs to judge, {len(done)} judged before")

    batch = pending[:limit]
    judged = pooling.judge_pool(batch, make_relevance_judge(settings), llm_path)
    typer.echo(f"{len(judged)} judged now, saved in {llm_path}")
    if len(batch) < len(pending):
        typer.echo(f"{len(pending) - len(batch)} pairs left: run again to finish before the review")
        return

    entries = pooling.review_sample([*done, *judged])
    pooling.write_review(entries, {q.id: q for q in questions}, chunks, review_path)
    relevant = sum(entry.verdict.relevant for entry in entries)
    typer.echo(
        f"review of {len(entries)} chunks ({relevant} found relevant by the LLM, "
        f"{len(entries) - relevant} sampled rejections) saved in {review_path}"
    )


@app.command("merge-pool")
def merge_pool(
    golden: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Golden set.")] = Path(
        "eval/golden.toml"
    ),
    judgments: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Rejected chunks.")
    ] = Path("eval/judgments.toml"),
) -> None:
    """Merge the filled review into the golden set and the judgments; twice changes nothing."""
    import tomllib

    from agentic_rag.config import get_settings
    from agentic_rag.evaluation import pool_merge, pooling
    from agentic_rag.evaluation.golden import GoldenSet
    from agentic_rag.factory import make_store

    review = pooling.load_review(pooling.POOLING_DIR / "review.toml")
    if errors := pooling.review_errors(review):
        typer.echo("\n".join(errors), err=True)
        raise typer.Exit(code=1)

    llm = pooling.load_pool_judgments(pooling.POOLING_DIR / "llm.jsonl")
    agreement = pooling.agreement(llm, review)
    false_rejections = agreement.false_rejections
    typer.echo(
        f"false rejections: {false_rejections.k}/{false_rejections.n}, "
        f"upper bound {false_rejections.high:.1%}: unchecked rejections "
        + ("kept" if agreement.rejections_kept else "NOT kept, only the reviewed ones count")
    )

    wordings = pool_merge.new_wordings(review)
    new_golden = pool_merge.add_wordings(golden.read_text(encoding="utf-8"), wordings)
    questions = GoldenSet.model_validate(tomllib.loads(new_golden)).questions

    texts = {c.key: c.chunk.text for c in make_store(get_settings()).all_chunks()}
    candidates = pool_merge.rejection_candidates(llm, review)
    useful = pool_merge.stating_a_fact(candidates, {q.id: q for q in questions}, texts)
    rejected = [pair for pair in candidates if pair not in useful]
    new_judgments = pool_merge.add_rejections(
        judgments.read_text(encoding="utf-8"), pool_merge.rejection_quotes(rejected, texts)
    )

    golden.write_text(new_golden, encoding="utf-8")  # last: a failure above leaves both files as is
    judgments.write_text(new_judgments, encoding="utf-8")
    typer.echo(f"{len(wordings)} wordings for the golden set, {len(rejected)} rejections")
    for question_id, key in useful:
        typer.echo(f"rejected by the LLM but states a fact, counted useful: {question_id} / {key}")
    for entry in review:
        if entry.verdict == "new_fact":
            typer.echo(f"new fact to add by hand: {entry.question_id} / {entry.chunk}")
    typer.echo("the golden set changed: run the evaluations again")


def _load_run(path: Path, hint: str) -> RunRecord:
    try:
        return load_run(path)
    except ValidationError as exc:
        raise typer.BadParameter(f"not a saved run: {exc}", param_hint=hint) from exc


def _load_pipeline(path: Path | None) -> PipelineConfig:
    """No file means the baseline."""
    if path is None:
        return PipelineConfig()
    try:
        return load_pipeline(path)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc), param_hint="--pipeline") from exc
