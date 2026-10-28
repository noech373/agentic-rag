import pytest

from agentic_rag.document import Chunk
from agentic_rag.evaluation.golden import Evidence, Question, Quote
from agentic_rag.evaluation.metrics import mrr_at_k, recall_at_k


def _chunk(text: str) -> Chunk:
    return Chunk(kind="text", text=text, path=())


def _question(*facts: tuple[str, ...]) -> Question:
    """A question whose facts are the given wordings, each one a document `gdpr`."""
    evidence = tuple(
        Evidence(any_of=tuple(Quote(document="gdpr", quote=wording) for wording in fact))
        for fact in facts
    )
    return Question(
        id="test",
        category="factual",
        origin="blind",
        question="?",
        answer="!",
        evidence=evidence,
    )


_NOTIFY = ("notify the supervisory authority without undue delay",)
_TIME = ("within 72 hours of becoming aware",)
_OTHER = ("nothing relevant here at all",)


def test_a_fact_found_in_the_first_k_chunks_counts() -> None:
    chunks = [_chunk("Controllers shall notify the supervisory authority without undue delay.")]

    assert recall_at_k(_question(_NOTIFY), chunks, k=1) == 1


def test_a_fact_found_after_k_does_not_count() -> None:
    chunks = [
        _chunk("Unrelated text."),
        _chunk("Notify the supervisory authority without undue delay."),
    ]

    assert recall_at_k(_question(_NOTIFY), chunks, k=1) == 0
    assert recall_at_k(_question(_NOTIFY), chunks, k=2) == 1


def test_recall_is_the_share_of_facts_covered() -> None:
    chunks = [_chunk("Notify the supervisory authority without undue delay.")]

    assert recall_at_k(_question(_NOTIFY, _TIME), chunks, k=1) == 0.5


def test_a_fact_in_several_chunks_counts_once() -> None:
    chunks = [
        _chunk("Notify the supervisory authority without undue delay."),
        _chunk("Notify the supervisory authority without undue delay, again."),
    ]

    assert recall_at_k(_question(_NOTIFY), chunks, k=2) == 1


def test_any_wording_of_a_fact_is_enough() -> None:
    wordings = ("the controller informs the authority", "notify the supervisory authority")
    chunks = [_chunk("Notify the supervisory authority without undue delay.")]

    assert recall_at_k(_question((wordings[0], wordings[1])), chunks, k=1) == 1


def test_a_fact_is_matched_as_whole_words() -> None:
    chunks = [_chunk("Within 172 hours of becoming aware.")]

    assert recall_at_k(_question(_TIME), chunks, k=1) == 0


def test_no_chunk_gives_zero() -> None:
    assert recall_at_k(_question(_NOTIFY, _TIME), [], k=5) == 0


def test_a_question_without_facts_has_no_recall() -> None:
    question = Question(
        id="refused",
        category="unanswerable",
        origin="blind",
        question="?",
        answer="!",
    )

    with pytest.raises(ValueError, match="no facts"):
        recall_at_k(question, [_chunk("anything")], k=1)


def test_mrr_is_one_over_the_rank_of_the_first_useful_chunk() -> None:
    chunks = [
        _chunk("Unrelated text."),
        _chunk("Notify the supervisory authority without undue delay."),
    ]

    assert mrr_at_k(_question(_NOTIFY), chunks, k=2) == 0.5


def test_mrr_is_one_when_the_first_chunk_is_useful() -> None:
    chunks = [_chunk("Notify the supervisory authority without undue delay.")]

    assert mrr_at_k(_question(_NOTIFY), chunks, k=1) == 1


def test_mrr_counts_a_chunk_useful_for_any_fact() -> None:
    chunks = [_chunk("Within 72 hours of becoming aware."), _chunk("Unrelated text.")]

    assert mrr_at_k(_question(_NOTIFY, _TIME), chunks, k=2) == 1


def test_mrr_is_zero_when_no_useful_chunk_is_in_the_top_k() -> None:
    chunks = [
        _chunk("Unrelated text."),
        _chunk("Notify the supervisory authority without undue delay."),
    ]

    assert mrr_at_k(_question(_NOTIFY), chunks, k=1) == 0


def test_mrr_of_a_question_without_facts_is_undefined() -> None:
    question = Question(
        id="refused",
        category="unanswerable",
        origin="blind",
        question="?",
        answer="!",
    )

    with pytest.raises(ValueError, match="no facts"):
        mrr_at_k(question, [_chunk("anything")], k=1)
