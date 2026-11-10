from pathlib import Path

import pytest

from agentic_rag.hyde import cached
from agentic_rag.hyde.base import AnswerWriter
from agentic_rag.hyde.cached import CachedAnswer, CachedAnswerWriter

QUESTION = "Our privacy person keeps blocking projects. Can we let her go for it?"
OTHER = "How often does a bank have to run a simulated attack on its live systems?"


class _Fake:
    """Answers with a new text at each call, as an LLM may, and can fail after some calls."""

    def __init__(self, name: str = "qwen-test", fail_after: int | None = None) -> None:
        self.name = name
        self.calls = 0
        self._fail_after = fail_after

    def write(self, question: str) -> str:
        if self.calls == self._fail_after:
            raise RuntimeError("the LLM failed")
        self.calls += 1
        return f"answer {self.calls}"


def _lines(path: Path) -> list[CachedAnswer]:
    return [CachedAnswer.model_validate_json(line) for line in path.read_text("utf-8").splitlines()]


def test_it_is_an_answer_writer(tmp_path: Path) -> None:
    writer: AnswerWriter = CachedAnswerWriter(_Fake(), tmp_path / "h.jsonl")

    assert writer.write(QUESTION) == "answer 1"


def test_the_same_question_is_not_asked_twice(tmp_path: Path) -> None:
    fake = _Fake()
    writer = CachedAnswerWriter(fake, tmp_path / "h.jsonl")

    writer.write(QUESTION)
    answer = writer.write(QUESTION)

    assert answer == "answer 1"
    assert fake.calls == 1


def test_a_new_run_reads_the_answers_of_the_previous_one(tmp_path: Path) -> None:
    """The run after pooling searches with the same answers as the run before it."""
    path = tmp_path / "h.jsonl"
    CachedAnswerWriter(_Fake(), path).write(QUESTION)
    fake = _Fake()

    answer = CachedAnswerWriter(fake, path).write(QUESTION)

    assert answer == "answer 1"
    assert fake.calls == 0


def test_another_question_asks_again(tmp_path: Path) -> None:
    fake = _Fake()
    writer = CachedAnswerWriter(fake, tmp_path / "h.jsonl")

    writer.write(QUESTION)

    assert writer.write(OTHER) == "answer 2"


def test_another_model_asks_again(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    CachedAnswerWriter(_Fake(name="qwen-test"), path).write(QUESTION)
    fake = _Fake(name="gpt-test")

    CachedAnswerWriter(fake, path).write(QUESTION)

    assert fake.calls == 1


def test_other_instructions_ask_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "h.jsonl"
    CachedAnswerWriter(_Fake(), path).write(QUESTION)
    monkeypatch.setattr(cached, "INSTRUCTIONS", "Answer in one word.")
    fake = _Fake()

    CachedAnswerWriter(fake, path).write(QUESTION)

    assert fake.calls == 1


def test_a_failure_keeps_the_answers_already_received_and_stores_nothing_for_it(
    tmp_path: Path,
) -> None:
    """An interrupted run resumes: only the questions not yet answered are asked again."""
    path = tmp_path / "h.jsonl"
    writer = CachedAnswerWriter(_Fake(fail_after=1), path)
    writer.write(QUESTION)

    with pytest.raises(RuntimeError):
        writer.write(OTHER)

    assert [line.question for line in _lines(path)] == [QUESTION]


def test_a_line_holds_the_model_the_question_and_the_answer(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"

    CachedAnswerWriter(_Fake(), path).write(QUESTION)

    (line,) = _lines(path)
    assert (line.model, line.question, line.answer) == ("qwen-test", QUESTION, "answer 1")
    assert len(line.key) == 64


def test_a_cached_answer_adds_no_line(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    writer = CachedAnswerWriter(_Fake(), path)

    writer.write(QUESTION)
    writer.write(QUESTION)

    assert len(_lines(path)) == 1


def test_it_keeps_the_name_of_its_writer(tmp_path: Path) -> None:
    assert CachedAnswerWriter(_Fake(name="qwen-test"), tmp_path / "h.jsonl").name == "qwen-test"
