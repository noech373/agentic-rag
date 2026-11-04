import tomllib

import pytest

from agentic_rag.evaluation.golden import GoldenSet
from agentic_rag.evaluation.pool_merge import NewWording, add_wordings, new_wordings
from agentic_rag.evaluation.pooling import ReviewEntry

_GOLDEN = """\
# A comment to keep.

[[questions]]
id = "first"
category = "factual"
origin = "blind"
question = "?"
answer = "!"

[[questions.evidence]]
any_of = [
  { document = "gdpr", quote = "the first fact of the first question" },
]

[[questions.evidence]]
any_of = [
  { document = "gdpr", quote = "the second fact of the first question" },
]

# --- A section to keep ---

[[questions]]
id = "second"
category = "factual"
origin = "blind"
question = "?"
answer = "!"

[[questions.evidence]]
any_of = [
  { document = "dora", quote = "the only fact of the second question" },
]
"""


def _evidence(golden: str, question: int, fact: int) -> list[tuple[str, str]]:
    parsed = GoldenSet.model_validate(tomllib.loads(golden))
    return [(q.document, q.quote) for q in parsed.questions[question].evidence[fact].any_of]


def _entry(key: str, verdict: str, fact: object = 0, quote: object = "") -> ReviewEntry:
    return ReviewEntry.model_validate(
        {
            "question_id": "first",
            "chunk": key,
            "question": "?",
            "answer": "!",
            "facts": ["a", "b"],
            "location": "GDPR",
            "text": "t",
            "verdict": verdict,
            "fact": fact,
            "quote": quote,
        }
    )


def test_a_quote_joins_the_wordings_of_its_fact() -> None:
    golden = add_wordings(_GOLDEN, [NewWording("first", 2, "edpb", "a new way to say it")])

    assert _evidence(golden, 0, 1) == [
        ("gdpr", "the second fact of the first question"),
        ("edpb", "a new way to say it"),
    ]
    assert len(_evidence(golden, 0, 0)) == 1


def test_the_right_question_gets_the_quote() -> None:
    golden = add_wordings(_GOLDEN, [NewWording("second", 1, "edpb", "a new way to say it")])

    assert _evidence(golden, 1, 0)[-1] == ("edpb", "a new way to say it")
    assert len(_evidence(golden, 0, 0)) == 1


def test_comments_and_layout_are_kept() -> None:
    golden = add_wordings(_GOLDEN, [NewWording("first", 1, "edpb", "a new way to say it")])

    added = '  { document = "edpb", quote = "a new way to say it" },\n'
    assert golden.replace(added, "") == _GOLDEN


def test_merging_twice_changes_nothing() -> None:
    wording = NewWording("first", 1, "edpb", "a new way to say it")
    once = add_wordings(_GOLDEN, [wording])

    assert add_wordings(once, [wording]) == once


def test_a_wording_differing_only_in_case_and_punctuation_is_not_added() -> None:
    wording = NewWording("first", 1, "edpb", "The FIRST fact, of the first question.")

    assert add_wordings(_GOLDEN, [wording]) == _GOLDEN


def test_a_quote_with_quotes_stays_valid_toml() -> None:
    quote = 'process so-called "sensitive" data'

    golden = add_wordings(_GOLDEN, [NewWording("first", 1, "cnil", quote)])

    assert _evidence(golden, 0, 0)[-1] == ("cnil", quote)


def test_an_existing_quote_with_quotes_is_recognized() -> None:
    wording = NewWording("first", 1, "cnil", 'process so-called "sensitive" data')
    once = add_wordings(_GOLDEN, [wording])

    assert add_wordings(once, [wording]) == once


@pytest.mark.parametrize(
    ("wording", "error"),
    [
        (NewWording("nobody", 1, "gdpr", "some words to add"), "not in the golden set"),
        (NewWording("first", 3, "gdpr", "some words to add"), "no fact 3"),
        (NewWording("second", 2, "gdpr", "some words to add"), "no fact 2"),
    ],
)
def test_an_unknown_target_is_an_error(wording: NewWording, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        add_wordings(_GOLDEN, [wording])


def test_only_the_relevant_chunks_give_wordings() -> None:
    review = [
        _entry("gdpr#1", "relevant", 1, "stated on\none line"),
        _entry("cnil-developer-guide-12#4", "relevant", [1, 2], ["for fact one", "for fact two"]),
        _entry("gdpr#2", "rejected"),
        _entry("gdpr#3", "new_fact", 0, "a fact the question misses"),
    ]

    assert new_wordings(review) == [
        NewWording("first", 1, "gdpr", "stated on one line"),
        NewWording("first", 1, "cnil-developer-guide-12", "for fact one"),
        NewWording("first", 2, "cnil-developer-guide-12", "for fact two"),
    ]
