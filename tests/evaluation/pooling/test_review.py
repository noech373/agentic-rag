import tomllib
from pathlib import Path

import pytest

from agentic_rag.document import Chunk, IndexedDocument, StoredChunk
from agentic_rag.evaluation.golden import Evidence, Question, Quote
from agentic_rag.evaluation.pooling.judge import StatedFact, Verdict
from agentic_rag.evaluation.pooling.review import (
    ReviewEntry,
    load_review,
    review_errors,
    review_sample,
    write_review,
)
from agentic_rag.evaluation.pooling.selection import PoolJudgment

_FACT = "notify the supervisory authority without undue delay"


def _question(id_: str) -> Question:
    return Question(
        id=id_,
        category="factual",
        origin="blind",
        question="?",
        answer="!",
        evidence=(Evidence(any_of=(Quote(document="gdpr", quote=_FACT),)),),
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


def test_the_review_checks_a_hundred_rejections_by_default() -> None:
    rejected = [_judgment("q", f"gdpr#{i}") for i in range(150)]

    assert len(review_sample(rejected)) == 100


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


_QUOTE = "shall notify the supervisory"
_TEXT = "Article 33. The controller shall notify the supervisory authority without undue delay."


def _entry(verdict: str, fact: object = 0, quote: object = "") -> ReviewEntry:
    return ReviewEntry.model_validate(
        {
            "question_id": "q",
            "chunk": "gdpr#1",
            "question": "?",
            "answer": "!",
            "facts": [_FACT, "the right to lodge a complaint"],
            "location": "GDPR",
            "text": _TEXT,
            "verdict": verdict,
            "fact": fact,
            "quote": quote,
        }
    )


def test_a_filled_review_loads_back(tmp_path: Path) -> None:
    path = tmp_path / "review.toml"
    _write(path)
    filled = path.read_text(encoding="utf-8").replace('verdict = ""', 'verdict = "rejected"')
    path.write_text(filled, encoding="utf-8")

    [entry] = load_review(path)

    assert (entry.question_id, entry.chunk, entry.verdict) == ("q", "gdpr#1", "rejected")
    assert entry.stated == ()


@pytest.mark.parametrize(
    "entry",
    [
        _entry("rejected"),
        _entry("relevant", 1, "shall notify the supervisory authority"),
        _entry("relevant", 1, "SHALL notify,\nthe supervisory   authority"),  # spacing, case
        _entry("new_fact", 0, "Article 33. The controller shall"),
    ],
)
def test_a_well_filled_entry_has_no_error(entry: ReviewEntry) -> None:
    assert review_errors([entry]) == []


def test_a_chunk_may_state_several_facts() -> None:
    quotes = ["shall notify the supervisory", "Article 33. The controller"]
    entry = _entry("relevant", [1, 2], quotes)

    assert review_errors([entry]) == []
    assert entry.stated == (
        StatedFact(fact=1, quote="shall notify the supervisory"),
        StatedFact(fact=2, quote="Article 33. The controller"),
    )


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        (_entry(""), "no verdict"),
        (_entry("rejected", 1, "shall notify the supervisory"), "a rejection has no fact"),
        (_entry("relevant", 3, "shall notify the supervisory"), "facts 1 to 2"),
        (_entry("relevant", 1, "shall inform the supervisory"), "not in the chunk"),
        (_entry("relevant", 1, "shall notify"), "2 words, not 4 to 25"),
        (_entry("relevant", 1, ""), "0 words"),
        (_entry("relevant", [1, 2], ["shall notify the supervisory"]), "2 facts but 1 quotes"),
        (_entry("relevant", [1, 1], ["shall notify the supervisory"] * 2), "given twice"),
        (_entry("new_fact", 1, "Article 33. The controller shall"), "keeps fact = 0"),
    ],
)
def test_a_badly_filled_entry_is_reported(entry: ReviewEntry, error: str) -> None:
    [message] = review_errors([entry])

    assert message.startswith("q / gdpr#1: ")
    assert error in message


def test_every_error_is_reported_at_once() -> None:
    other = _entry("relevant", 9, "shall notify").model_copy(update={"chunk": "gdpr#2"})

    errors = review_errors([_entry(""), other])

    assert len(errors) == 3  # no verdict; fact 9 out of range and too short a quote


def test_a_chunk_reviewed_twice_is_reported() -> None:
    assert review_errors([_entry("rejected"), _entry("rejected")]) == ["q / gdpr#1: reviewed twice"]


def test_an_unknown_verdict_does_not_load() -> None:
    with pytest.raises(ValueError, match="verdict"):
        _entry("relevant!")
