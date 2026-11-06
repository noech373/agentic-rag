import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_rag.cli import _load_pipeline, app  # pyright: ignore[reportPrivateUsage]
from agentic_rag.document import Chunk, IndexedDocument, StoredChunk
from agentic_rag.evaluation import record
from agentic_rag.evaluation.pooling.judge import StatedFact, Verdict
from agentic_rag.evaluation.pooling.selection import POOLING_DIR, PoolJudgment, load_pool_judgments
from agentic_rag.evaluation.record import RunRecord, file_sha256, load_run, save_run
from agentic_rag.evaluation.runner import QuestionResult
from agentic_rag.pipeline_config import PipelineConfig

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No .env and no keys: a command that reads the settings fails on them."""
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "POSTGRES_USER", "POSTGRES_PASSWORD"):
        monkeypatch.delenv(name, raising=False)


def test_no_pipeline_file_means_the_baseline() -> None:
    assert _load_pipeline(None) == PipelineConfig()


def test_a_pipeline_file_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "variant.toml"
    path.write_text('retriever = "vector"\n')

    assert _load_pipeline(path) == PipelineConfig(retriever="vector")


def test_a_bad_pipeline_file_is_a_usage_error_before_the_settings_are_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "variant.toml"
    path.write_text("rerankr = true\n")

    result = runner.invoke(app, ["ask", "What is a personal data breach?", "--pipeline", str(path)])

    assert result.exit_code == 2  # usage error, not a traceback
    assert "rerankr" in result.output
    assert "api_key" not in result.output


def test_a_missing_pipeline_file_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ask", "question", "--pipeline", str(tmp_path / "nope.toml")])

    assert result.exit_code == 2


def test_a_directory_is_not_a_pipeline_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ask", "question", "--pipeline", str(tmp_path)])

    assert result.exit_code == 2


def _run(score: float, **provenance: object) -> RunRecord:
    recall = {f"recall@{k}": score for k in (5, 8, 20, 50)}
    results = tuple(
        QuestionResult(
            question_id=f"q{i}",
            category="factual",
            recall=recall,
            mrr=score,
            judged=score,
            retrieved=(),
        )
        for i in range(3)
    )
    run = RunRecord(
        golden_sha256="9f2c",
        judgments_sha256="3d1e",
        git_sha="e87c5ad0000",
        dirty=False,
        config={"retriever": "vector"},
        results=results,
    )
    return run.model_copy(update=provenance)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[PipelineConfig]:
    """Replace the retrieval with a fixed run of score 1; return the configs it was asked to run."""
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "golden.toml").write_text("")
    (tmp_path / "eval" / "judgments.toml").write_text("")
    monkeypatch.setattr("agentic_rag.config.get_settings", lambda: None)
    configs: list[PipelineConfig] = []

    def fake_record_run(
        golden: Path,
        judgments: Path,
        make_retriever: object,
        pipeline: PipelineConfig,
        repo: Path,
    ) -> RunRecord:
        configs.append(pipeline)
        return _run(1.0)

    monkeypatch.setattr(record, "record_run", fake_record_run)
    return configs


def _variant(tmp_path: Path, name: str = "variant") -> str:
    path = tmp_path / f"{name}.toml"
    path.write_text('retriever = "vector"\n')
    return str(path)


def test_eval_without_pipeline_runs_and_saves_the_baseline(recorded: list[PipelineConfig]) -> None:
    result = runner.invoke(app, ["eval"])

    assert result.exit_code == 0, result.output
    assert recorded == [PipelineConfig()]
    assert load_run(record.RESULTS_DIR / "baseline.json") == _run(1.0)


def test_eval_shows_the_means_of_the_run(recorded: list[PipelineConfig]) -> None:
    result = runner.invoke(app, ["eval"])

    assert "| metric | baseline [95% CI] |" in result.stdout
    assert "| recall@8 | 1.00 [1.00, 1.00] |" in result.stdout


def test_eval_of_a_variant_is_saved_under_its_file_name(
    recorded: list[PipelineConfig], tmp_path: Path
) -> None:
    result = runner.invoke(app, ["eval", "--pipeline", _variant(tmp_path, "bm25")])

    assert result.exit_code == 0, result.output
    assert recorded == [PipelineConfig(retriever="vector")]
    assert (record.RESULTS_DIR / "bm25.json").is_file()
    assert "| metric | bm25 [95% CI] |" in result.stdout


def test_eval_does_not_compare_with_a_saved_baseline(
    recorded: list[PipelineConfig], tmp_path: Path
) -> None:
    save_run(_run(0.0), record.RESULTS_DIR / "baseline.json")

    result = runner.invoke(app, ["eval", "--pipeline", _variant(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "difference" not in result.stdout
    assert load_run(record.RESULTS_DIR / "baseline.json") == _run(0.0)  # left as it was


def test_rerunning_a_variant_replaces_its_run(recorded: list[PipelineConfig]) -> None:
    save_run(_run(0.0), record.RESULTS_DIR / "baseline.json")

    result = runner.invoke(app, ["eval"])

    assert result.exit_code == 0, result.output
    assert load_run(record.RESULTS_DIR / "baseline.json") == _run(1.0)


def test_a_bad_pipeline_file_stops_eval_before_any_run(
    recorded: list[PipelineConfig], tmp_path: Path
) -> None:
    path = tmp_path / "variant.toml"
    path.write_text("rerankr = true\n")

    result = runner.invoke(app, ["eval", "--pipeline", str(path)])

    assert result.exit_code == 2
    assert recorded == []


class _NoIngester:
    def __init__(self, *_args: object) -> None:
        pass

    def run(self, *_args: object, **_kwargs: object) -> list[object]:
        return []


@pytest.fixture
def ingest_corpora(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The corpus of each store `ingest` asks for, with nothing parsed, embedded or stored."""
    corpora: list[str] = []

    def make_store(_settings: object, corpus: str = "public") -> None:
        corpora.append(corpus)

    def load_manifest(_path: Path) -> _NoDocuments:
        return _NoDocuments()

    monkeypatch.setattr("agentic_rag.config.get_settings", lambda: None)
    monkeypatch.setattr("agentic_rag.factory.make_store", make_store)
    monkeypatch.setattr("agentic_rag.factory.make_embedder", _offline(None))
    monkeypatch.setattr("agentic_rag.ingestion.pipeline.Ingester", _NoIngester)
    monkeypatch.setattr("agentic_rag.cli.load_manifest", load_manifest)
    return corpora


class _NoDocuments:
    documents = ()


def test_ingest_writes_to_the_public_corpus_by_default(ingest_corpora: list[str]) -> None:
    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 0, result.output
    assert ingest_corpora == ["public"]


def test_ingest_writes_to_the_corpus_it_is_given(ingest_corpora: list[str]) -> None:
    result = runner.invoke(app, ["ingest", "--corpus", "titles"])

    assert result.exit_code == 0, result.output
    assert ingest_corpora == ["titles"]


@pytest.mark.parametrize("name", ["Titles", "ti-tles", "a.b", "x; DROP SCHEMA public CASCADE"])
def test_ingest_refuses_a_corpus_name_unsafe_in_sql(ingest_corpora: list[str], name: str) -> None:
    result = runner.invoke(app, ["ingest", "--corpus", name])

    assert result.exit_code == 2
    assert ingest_corpora == []


def test_eval_needs_the_golden_set(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "--golden", str(tmp_path / "nope.toml")])

    assert result.exit_code == 2


def test_eval_needs_the_judgments(recorded: list[PipelineConfig], tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "--judgments", str(tmp_path / "judgment.toml")])

    assert result.exit_code == 2
    assert recorded == []


def _saved(tmp_path: Path, name: str, run: RunRecord) -> str:
    path = tmp_path / f"{name}.json"
    save_run(run, path)
    return str(path)


def test_compare_shows_the_difference_between_two_saved_runs(tmp_path: Path) -> None:
    before = _saved(tmp_path, "bm25", _run(0.0))
    after = _saved(tmp_path, "bm25-rerank", _run(1.0))

    result = runner.invoke(app, ["compare", before, after])

    assert result.exit_code == 0, result.output
    assert "| metric | bm25 | bm25-rerank | difference [95% CI] |" in result.stdout
    assert "| recall@8 | 0.00 | 1.00 | **+1.00 [+1.00, +1.00]** |" in result.stdout
    assert result.stderr == ""


def test_compare_warns_on_stderr(tmp_path: Path) -> None:
    before = _saved(tmp_path, "baseline", _run(0.0, git_sha="0842afe0000", dirty=True))
    after = _saved(tmp_path, "bm25", _run(1.0))

    result = runner.invoke(app, ["compare", before, after])

    assert result.exit_code == 0, result.output
    assert "difference [95% CI]" in result.stdout
    assert "rerun the baseline" in result.stderr
    assert "not for the README" in result.stderr


def test_compare_refuses_runs_on_different_golden_sets(tmp_path: Path) -> None:
    before = _saved(tmp_path, "baseline", _run(0.0, golden_sha256="1b7a"))
    after = _saved(tmp_path, "bm25", _run(1.0))

    result = runner.invoke(app, ["compare", before, after])

    assert result.exit_code == 1
    assert "golden" in result.stderr
    assert result.stdout == ""


def test_compare_refuses_a_file_that_is_not_a_run(tmp_path: Path) -> None:
    before = _saved(tmp_path, "baseline", _run(0.0))
    other = tmp_path / "other.json"
    other.write_text("{}")

    result = runner.invoke(app, ["compare", before, str(other)])

    assert result.exit_code == 2
    assert "not a saved run" in result.output


def test_compare_needs_existing_files(tmp_path: Path) -> None:
    result = runner.invoke(app, ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")])

    assert result.exit_code == 2


_POOL_GOLDEN = """
[[questions]]
id = "q0"
category = "factual"
origin = "blind"
question = "When to notify?"
answer = "Without undue delay."
[[questions.evidence]]
any_of = [{ document = "gdpr", quote = "without undue delay" }]
"""


class _PoolStore:
    def all_chunks(self) -> list[StoredChunk]:
        document = IndexedDocument(
            id="gdpr",
            title="GDPR",
            short_title="GDPR",
            source_url=None,
            regulations=(),
            groups=("public",),
            sha256="0",
            embedding_model="m",
        )
        texts = ["Notify without undue delay.", "A relevant passage.", "Unrelated."]
        return [
            StoredChunk(document=document, position=i, chunk=Chunk(kind="text", text=t, path=()))
            for i, t in enumerate(texts)
        ]


class _PoolJudge:
    model = "gpt-test"

    def judge(self, question: object, chunk: StoredChunk) -> Verdict:
        facts = (StatedFact(fact=1, quote="x"),) if "relevant" in chunk.chunk.text else ()
        return Verdict(reason="Because.", facts=facts)


def _offline[T](value: T) -> Callable[[object], T]:
    """A factory that ignores the settings and returns `value`."""
    return lambda _settings: value


@pytest.fixture
def pooled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A golden set, its judgments, an offline store and judge; returns the golden set's path."""
    golden = tmp_path / "eval" / "golden.toml"
    golden.parent.mkdir()
    golden.write_text(_POOL_GOLDEN)
    (tmp_path / "eval" / "judgments.toml").write_text("")
    monkeypatch.setattr("agentic_rag.config.get_settings", lambda: None)
    monkeypatch.setattr("agentic_rag.factory.make_store", _offline(_PoolStore()))
    monkeypatch.setattr("agentic_rag.factory.make_relevance_judge", _offline(_PoolJudge()))
    return golden


def _pool_run(golden: Path) -> RunRecord:
    run = _run(0.0, golden_sha256=file_sha256(golden))
    retrieved = ("gdpr#0", "gdpr#1", "gdpr#2")
    q0 = run.results[0].model_copy(update={"question_id": "q0", "retrieved": retrieved})
    return run.model_copy(update={"results": (q0,)})


def test_pool_judges_the_unjudged_chunks_and_writes_the_review(
    pooled: Path, tmp_path: Path
) -> None:
    run = _saved(tmp_path, "baseline", _pool_run(pooled))

    result = runner.invoke(app, ["pool", run])

    assert result.exit_code == 0, result.output
    assert "2 pairs to judge, 0 judged before" in result.stdout  # gdpr#0 states the fact
    assert "2 judged now" in result.stdout
    assert "review of 2 chunks (1 found relevant by the LLM, 1 sampled rejections)" in result.stdout
    assert len(load_pool_judgments(POOLING_DIR / "llm.jsonl")) == 2
    review = tomllib.loads((POOLING_DIR / "review.toml").read_text())
    assert {entry["chunk"] for entry in review["chunks"]} == {"gdpr#1", "gdpr#2"}


def test_pool_with_a_limit_stops_before_the_review_and_resumes(
    pooled: Path, tmp_path: Path
) -> None:
    run = _saved(tmp_path, "baseline", _pool_run(pooled))

    first = runner.invoke(app, ["pool", run, "--limit", "1"])

    assert first.exit_code == 0, first.output
    assert "1 judged now" in first.stdout
    assert "1 pairs left" in first.stdout
    assert not (POOLING_DIR / "review.toml").exists()

    second = runner.invoke(app, ["pool", run])

    assert "1 pairs to judge, 1 judged before" in second.stdout
    assert len(load_pool_judgments(POOLING_DIR / "llm.jsonl")) == 2
    assert (POOLING_DIR / "review.toml").is_file()


def test_pool_never_overwrites_a_review(pooled: Path, tmp_path: Path) -> None:
    run = _saved(tmp_path, "baseline", _pool_run(pooled))
    POOLING_DIR.mkdir(parents=True)
    (POOLING_DIR / "review.toml").write_text("# my review\n")

    result = runner.invoke(app, ["pool", run])

    assert result.exit_code == 1
    assert "exists" in result.stderr
    assert (POOLING_DIR / "review.toml").read_text() == "# my review\n"
    assert not (POOLING_DIR / "llm.jsonl").exists()  # nothing was judged


def test_pool_refuses_runs_on_another_golden_set(pooled: Path, tmp_path: Path) -> None:
    run = _saved(tmp_path, "bm25", _run(0.0, golden_sha256="1b7a"))

    result = runner.invoke(app, ["pool", run])

    assert result.exit_code == 2
    assert "another golden set" in result.output
    assert not POOLING_DIR.exists()


_REVIEW_FACT = "the controller shall notify the authority"


def _review_entry(key: str, verdict: str, fact: int, quote: str) -> str:
    return f"""
[[chunks]]
question_id = "q0"
chunk = "{key}"
question = "?"
answer = "!"
facts = ["a"]
location = "GDPR"
text = "{_REVIEW_FACT} of a breach, says this chunk {key}."
verdict = "{verdict}"
fact = {fact}
quote = "{quote}"
"""


@pytest.fixture
def merged(pooled: Path) -> Path:
    """A filled review of gdpr#1 and gdpr#2, the LLM's verdicts on them and on gdpr#3."""
    pooled.write_text(
        _POOL_GOLDEN.replace(
            'any_of = [{ document = "gdpr", quote = "without undue delay" }]',
            'any_of = [\n  { document = "gdpr", quote = "without undue delay" },\n]',
        )
    )
    POOLING_DIR.mkdir(parents=True)
    (POOLING_DIR / "review.toml").write_text(
        _review_entry("gdpr#1", "relevant", 1, _REVIEW_FACT)
        + _review_entry("gdpr#2", "rejected", 0, "")
    )
    verdicts = [
        PoolJudgment(
            question_id="q0",
            chunk=key,
            model="m",
            verdict=Verdict(reason="r", facts=(StatedFact(fact=1, quote="x"),) if relevant else ()),
        )
        for key, relevant in (("gdpr#1", True), ("gdpr#2", True), ("gdpr#3", False))
    ]
    (POOLING_DIR / "llm.jsonl").write_text("".join(v.model_dump_json() + "\n" for v in verdicts))
    return pooled


def test_merge_pool_writes_the_wordings_and_the_rejections(merged: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["merge-pool"])

    assert result.exit_code == 0, result.output
    assert "false rejections: 0/0" in result.stdout
    assert "1 wordings for the golden set" in result.stdout
    assert _REVIEW_FACT in merged.read_text()
    assert "run the evaluations again" in result.stdout
    rejected = tomllib.loads((tmp_path / "eval" / "judgments.toml").read_text())
    assert [quote["document"] for quote in rejected["q0"]] == ["gdpr"]


def test_merge_pool_twice_changes_nothing(merged: Path, tmp_path: Path) -> None:
    runner.invoke(app, ["merge-pool"])
    golden = merged.read_text()
    judgments = (tmp_path / "eval" / "judgments.toml").read_text()

    result = runner.invoke(app, ["merge-pool"])

    assert result.exit_code == 0, result.output
    assert merged.read_text() == golden
    assert (tmp_path / "eval" / "judgments.toml").read_text() == judgments


def test_merge_pool_refuses_an_unfinished_review(merged: Path) -> None:
    before = merged.read_text()
    (POOLING_DIR / "review.toml").write_text(_review_entry("gdpr#1", "", 0, ""))

    result = runner.invoke(app, ["merge-pool"])

    assert result.exit_code == 1
    assert "q0 / gdpr#1: no verdict" in result.stderr
    assert merged.read_text() == before
