import pytest

from agentic_rag.document import Chunk
from agentic_rag.evaluation.comparison import IncomparableRunsError
from agentic_rag.evaluation.golden import Evidence, Question, Quote
from agentic_rag.evaluation.pooling import chunks_to_judge, pool
from agentic_rag.evaluation.record import RunRecord
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
