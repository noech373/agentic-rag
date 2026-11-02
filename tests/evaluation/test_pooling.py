import tomllib
from pathlib import Path

import pytest

from agentic_rag.document import Chunk
from agentic_rag.evaluation.comparison import IncomparableRunsError
from agentic_rag.evaluation.golden import Evidence, Question, Quote
from agentic_rag.evaluation.pooling import (
    PoolJudgment,
    chunks_to_judge,
    judge_pool,
    load_pool_judgments,
    not_yet_judged,
    pool,
    review_sample,
    write_review,
)
from agentic_rag.evaluation.record import RunRecord
from agentic_rag.evaluation.relevance_judge import StatedFact, Verdict
from agentic_rag.evaluation.runner import QuestionResult
from agentic_rag.storage.base import IndexedDocument, StoredChunk

_FACT = "notify the supervisory authority without undue delay"
_REJECTED = "the right to lodge a complaint"


def _question(id_: str) -> Question:
    return Question(
        id=id_,
        category="factual",
        origin="blind",
        question="?",
        answer="!",
        evidence=(Evidence(any_of=(Quote(document="gdpr", quote=_FACT),)),),
    )


def _result(question_id: str, *retrieved: str) -> QuestionResult:
    return QuestionResult(
        question_id=question_id,
        category="factual",
        recall={},
        mrr=0,
        judged=0,
        retrieved=retrieved,
    )


def _run(*results: QuestionResult, golden_sha256: str = "9f2c") -> RunRecord:
    return RunRecord(
        golden_sha256=golden_sha256,
        judgments_sha256="3d1e",
        git_sha="e87c5ad",
        dirty=False,
        config={},
        results=results,
    )


def _stored(key: str, text: str = "Unrelated text.") -> StoredChunk:
    document_id, position = key.split("#")
    document = IndexedDocument(
        id=document_id,
        title=document_id,
        short_title=document_id.upper(),
        source_url=None,
        regulations=(),
        groups=("public",),
        sha256="0",
        embedding_model="m",
    )
    return StoredChunk(
        document=document,
        position=int(position),
        chunk=Chunk(kind="text", text=text, path=()),
    )


def _index(*chunks: StoredChunk) -> dict[str, StoredChunk]:
    return {chunk.key: chunk for chunk in chunks}


def _pairs(to_judge: list[tuple[Question, StoredChunk]]) -> list[tuple[str, str]]:
    return [(question.id, chunk.key) for question, chunk in to_judge]


def test_the_pool_merges_the_runs_without_duplicates_in_the_order_first_seen() -> None:
    vector = _run(_result("q", "gdpr#1", "gdpr#2"))
    bm25 = _run(_result("q", "gdpr#3", "gdpr#1"))

    assert pool("q", [vector, bm25]) == ["gdpr#1", "gdpr#2", "gdpr#3"]


def test_the_pool_takes_only_the_first_chunks_of_each_run() -> None:
    vector = _run(_result("q", "gdpr#1", "gdpr#2", "gdpr#3"))
    bm25 = _run(_result("q", "gdpr#4", "gdpr#5"))

    assert pool("q", [vector, bm25], depth=1) == ["gdpr#1", "gdpr#4"]


def test_the_pool_is_per_question() -> None:
    run = _run(_result("q1", "gdpr#1"), _result("q2", "gdpr#2"))

    assert pool("q2", [run]) == ["gdpr#2"]


def test_a_question_absent_from_the_runs_has_an_empty_pool() -> None:
    assert pool("held-out", [_run(_result("q", "gdpr#1"))]) == []


def test_unjudged_pooled_chunks_are_to_judge() -> None:
    run = _run(_result("q", "gdpr#1", "gdpr#2"))

    to_judge = chunks_to_judge(
        [_question("q")], [run], {}, _index(_stored("gdpr#1"), _stored("gdpr#2"))
    )

    assert _pairs(to_judge) == [("q", "gdpr#1"), ("q", "gdpr#2")]


def test_a_chunk_that_states_a_fact_is_already_judged() -> None:
    run = _run(_result("q", "gdpr#1", "gdpr#2"))
    chunks = _index(_stored("gdpr#1", f"Controllers shall {_FACT}."), _stored("gdpr#2"))

    assert _pairs(chunks_to_judge([_question("q")], [run], {}, chunks)) == [("q", "gdpr#2")]


def test_a_chunk_with_a_rejected_quote_is_already_judged_for_its_question_only() -> None:
    run = _run(_result("q1", "gdpr#1"), _result("q2", "gdpr#1"))
    chunks = _index(_stored("gdpr#1", f"It sets out {_REJECTED}."))
    judgments = {"q1": (Quote(document="gdpr", quote=_REJECTED),)}

    to_judge = chunks_to_judge([_question("q1"), _question("q2")], [run], judgments, chunks)

    assert _pairs(to_judge) == [("q2", "gdpr#1")]


def test_the_same_chunk_is_to_judge_once_per_question() -> None:
    vector = _run(_result("q1", "gdpr#1"), _result("q2", "gdpr#1"))
    bm25 = _run(_result("q1", "gdpr#1"))

    to_judge = chunks_to_judge(
        [_question("q1"), _question("q2")], [vector, bm25], {}, _index(_stored("gdpr#1"))
    )

    assert _pairs(to_judge) == [("q1", "gdpr#1"), ("q2", "gdpr#1")]


def test_a_chunk_missing_from_the_index_stops_the_pooling() -> None:
    run = _run(_result("q", "gdpr#9"))

    with pytest.raises(ValueError, match=r"gdpr#9 is not in the index"):
        chunks_to_judge([_question("q")], [run], {}, _index(_stored("gdpr#1")))


def test_runs_on_different_golden_sets_cannot_be_pooled() -> None:
    runs = [_run(_result("q", "gdpr#1")), _run(_result("q", "gdpr#1"), golden_sha256="1b7a")]

    with pytest.raises(IncomparableRunsError, match="golden"):
        chunks_to_judge([_question("q")], runs, {}, _index(_stored("gdpr#1")))


class _FakeJudge:
    """Relevant when the chunk says "relevant"; fails on the chunk at position `fail_at`."""

    model = "fake-judge"

    def __init__(self, fail_at: int | None = None) -> None:
        self.fail_at = fail_at

    def judge(self, question: Question, chunk: StoredChunk) -> Verdict:
        if chunk.position == self.fail_at:
            raise RuntimeError("API down")
        if "relevant" in chunk.chunk.text:
            return Verdict(reason="States it.", facts=(StatedFact(fact=1, quote="relevant"),))
        return Verdict(reason="Same topic only.", facts=())


def _verdict_of(judgments: list[PoolJudgment], key: str) -> bool:
    [judgment] = [j for j in judgments if j.chunk == key]
    return judgment.verdict.relevant


def test_each_pair_gets_one_verdict_saved_to_the_file(tmp_path: Path) -> None:
    path = tmp_path / "pooling" / "llm.jsonl"
    to_judge = [
        (_question("q"), _stored("gdpr#1", "relevant text")),
        (_question("q"), _stored("gdpr#2")),
    ]

    judged = judge_pool(to_judge, _FakeJudge(), path, workers=2)

    saved = load_pool_judgments(path)
    assert sorted(saved, key=lambda j: j.chunk) == sorted(judged, key=lambda j: j.chunk)
    assert {(j.question_id, j.chunk, j.model) for j in saved} == {
        ("q", "gdpr#1", "fake-judge"),
        ("q", "gdpr#2", "fake-judge"),
    }
    assert _verdict_of(saved, "gdpr#1")
    assert not _verdict_of(saved, "gdpr#2")


def test_verdicts_are_appended_to_those_already_saved(tmp_path: Path) -> None:
    path = tmp_path / "llm.jsonl"

    judge_pool([(_question("q"), _stored("gdpr#1"))], _FakeJudge(), path)
    judge_pool([(_question("q"), _stored("gdpr#2"))], _FakeJudge(), path)

    assert [j.chunk for j in load_pool_judgments(path)] == ["gdpr#1", "gdpr#2"]


def test_a_failed_call_keeps_the_verdicts_received_before_and_raises(tmp_path: Path) -> None:
    path = tmp_path / "llm.jsonl"
    to_judge = [(_question("q"), _stored(f"gdpr#{i}")) for i in (1, 2, 3)]

    with pytest.raises(RuntimeError, match="API down"):
        judge_pool(to_judge, _FakeJudge(fail_at=2), path, workers=1)

    assert [j.chunk for j in load_pool_judgments(path)] == ["gdpr#1"]


def test_nothing_is_saved_before_the_first_pooling(tmp_path: Path) -> None:
    assert load_pool_judgments(tmp_path / "llm.jsonl") == []


def test_pairs_the_llm_already_judged_are_not_judged_again() -> None:
    done = [
        PoolJudgment(
            question_id="q1", chunk="gdpr#1", model="m", verdict=Verdict(reason="", facts=())
        )
    ]
    to_judge = [
        (_question("q1"), _stored("gdpr#1")),
        (_question("q1"), _stored("gdpr#2")),
        (_question("q2"), _stored("gdpr#1")),  # same chunk, other question: still to judge
    ]

    assert _pairs(not_yet_judged(to_judge, done)) == [("q1", "gdpr#2"), ("q2", "gdpr#1")]


def test_resuming_after_a_failure_judges_only_what_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "llm.jsonl"
    to_judge = [(_question("q"), _stored(f"gdpr#{i}")) for i in (1, 2, 3)]
    with pytest.raises(RuntimeError):
        judge_pool(to_judge, _FakeJudge(fail_at=2), path, workers=1)

    judge_pool(not_yet_judged(to_judge, load_pool_judgments(path)), _FakeJudge(), path)

    assert sorted(j.chunk for j in load_pool_judgments(path)) == ["gdpr#1", "gdpr#2", "gdpr#3"]


def _judgment(question_id: str, key: str, *, relevant: bool = False) -> PoolJudgment:
    facts = (StatedFact(fact=1, quote="notify"),) if relevant else ()
    return PoolJudgment(
        question_id=question_id,
        chunk=key,
        model="m",
        verdict=Verdict(reason="Because it says so.", facts=facts),
    )


def test_the_review_has_every_relevant_verdict_and_a_sample_of_rejections() -> None:
    relevant = [_judgment("q", f"gdpr#{i}", relevant=True) for i in range(3)]
    rejected = [_judgment("q", f"gdpr#{i}") for i in range(10, 30)]

    entries = review_sample([*rejected, *relevant], sample=5)

    assert sum(e.verdict.relevant for e in entries) == 3
    assert sum(not e.verdict.relevant for e in entries) == 5


def test_the_sample_is_the_same_whatever_the_order_of_the_verdicts() -> None:
    rejected = [_judgment("q", f"gdpr#{i}") for i in range(20)]

    assert review_sample(rejected, sample=5) == review_sample(list(reversed(rejected)), sample=5)


def test_the_sample_takes_all_rejections_when_there_are_fewer() -> None:
    rejected = [_judgment("q", "gdpr#1"), _judgment("q", "gdpr#2")]

    assert len(review_sample(rejected, sample=50)) == 2


def test_the_review_order_does_not_tell_the_verdict() -> None:
    entries = [
        _judgment("b", "gdpr#1", relevant=True),
        _judgment("a", "gdpr#2"),
        _judgment("a", "gdpr#1", relevant=True),
    ]

    assert [(e.question_id, e.chunk) for e in review_sample(entries)] == [
        ("a", "gdpr#1"),
        ("a", "gdpr#2"),
        ("b", "gdpr#1"),
    ]


def _write(path: Path, text: str = "Notify the authority.") -> dict[str, object]:
    chunk = _stored("gdpr#1", text)
    write_review(
        [_judgment("q", "gdpr#1", relevant=True)], {"q": _question("q")}, {chunk.key: chunk}, path
    )
    [entry] = tomllib.loads(path.read_text(encoding="utf-8"))["chunks"]
    return entry


def test_the_review_shows_what_to_judge_and_not_the_llm_verdict(tmp_path: Path) -> None:
    path = tmp_path / "pooling" / "review.toml"

    entry = _write(path)

    assert entry == {
        "question_id": "q",
        "chunk": "gdpr#1",
        "question": "?",
        "answer": "!",
        "facts": [_FACT],
        "location": "GDPR",
        "text": "Notify the authority.",
        "verdict": "",
        "fact": 0,
        "quote": "",
        "note": "",
    }
    assert "Because" not in path.read_text(encoding="utf-8")  # the LLM's reason


def test_the_facts_list_every_wording_of_each_fact(tmp_path: Path) -> None:
    question = _question("q").model_copy(
        update={
            "evidence": (
                Evidence(
                    any_of=(Quote(document="gdpr", quote="a"), Quote(document="x", quote="b"))
                ),
                Evidence(any_of=(Quote(document="gdpr", quote="c"),)),
            )
        }
    )
    chunk = _stored("gdpr#1")
    path = tmp_path / "review.toml"

    write_review([_judgment("q", "gdpr#1")], {"q": question}, {chunk.key: chunk}, path)

    [entry] = tomllib.loads(path.read_text(encoding="utf-8"))["chunks"]
    assert entry["facts"] == ["a | b", "c"]


_TRICKY = "Line 1 with \"quotes\" and a \\ backslash.\nLine 2 with ''' and \t a tab, ✔ and é."


@pytest.mark.parametrize(
    "text",
    ["A plain chunk.\nOn two lines.", _TRICKY, "Ends with a quote'", "Bell \x07 and DEL \x7f"],
)
def test_the_review_is_toml_that_keeps_every_text(tmp_path: Path, text: str) -> None:
    entry = _write(tmp_path / "review.toml", text)

    assert entry["text"] == text


def test_a_plain_text_stays_readable_as_written(tmp_path: Path) -> None:
    path = tmp_path / "review.toml"

    _write(path, "First line.\nSecond line.")

    assert "text = '''\nFirst line.\nSecond line.'''" in path.read_text(encoding="utf-8")


def test_an_existing_review_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "review.toml"
    path.write_text("# my review, half done\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="rename or delete it"):
        _write(path)

    assert path.read_text(encoding="utf-8") == "# my review, half done\n"
