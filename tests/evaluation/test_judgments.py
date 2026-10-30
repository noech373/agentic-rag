"""Judgments load as rejected quotes per question, and the real file matches the golden set."""

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.evaluation.golden import Quote, load_golden_set
from agentic_rag.evaluation.judgments import load_judgments

_ROOT = Path(__file__).parents[2]


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "judgments.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_rejected_quotes_are_loaded_per_question(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
chatbot-disclose-ai = [
  { document = "ai-act", quote = "Given the major impact that AI can have on society" },
  { document = "nis2", quote = "the relevant entities shall" },
]
ip-addresses-personal-data = [{ document = "gdpr", quote = "the purpose of this regulation" }]
""",
    )

    assert load_judgments(path) == {
        "chatbot-disclose-ai": (
            Quote(document="ai-act", quote="Given the major impact that AI can have on society"),
            Quote(document="nis2", quote="the relevant entities shall"),
        ),
        "ip-addresses-personal-data": (
            Quote(document="gdpr", quote="the purpose of this regulation"),
        ),
    }


def test_an_empty_file_has_no_judgments(tmp_path: Path) -> None:
    assert load_judgments(_write(tmp_path, "# nothing judged yet\n")) == {}


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_judgments(tmp_path / "judgment.toml")


def test_a_question_given_twice_is_an_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
chatbot-disclose-ai = [{ document = "ai-act", quote = "first judgment of this question" }]
chatbot-disclose-ai = [{ document = "ai-act", quote = "second judgment of this question" }]
""",
    )

    with pytest.raises(tomllib.TOMLDecodeError):
        load_judgments(path)


@pytest.mark.parametrize(
    "entry",
    [
        '{ quote = "a quote without its document" }',
        '{ document = "gdpr", quote = "a quote", judge = "llm" }',
        '"a bare string"',
    ],
)
def test_a_malformed_quote_is_an_error(tmp_path: Path, entry: str) -> None:
    path = _write(tmp_path, f"chatbot-disclose-ai = [{entry}]\n")

    with pytest.raises(ValidationError):
        load_judgments(path)


def test_the_real_judgments_are_about_golden_questions() -> None:
    golden_ids = {q.id for q in load_golden_set(_ROOT / "eval" / "golden.toml").questions}

    unknown = load_judgments(_ROOT / "eval" / "judgments.toml").keys() - golden_ids

    assert not unknown, f"not in golden.toml: {sorted(unknown)}"
