import tomllib

import pytest

from agentic_rag.evaluation.golden import Evidence, GoldenSet, Question, Quote, contains, normalize
from agentic_rag.evaluation.pooling.judge import StatedFact, Verdict
from agentic_rag.evaluation.pooling.merge import (
    NewWording,
    add_rejections,
    add_wordings,
    new_wordings,
    rejection_candidates,
    rejection_quotes,
    stating_a_fact,
)
from agentic_rag.evaluation.pooling.selection import PoolJudgment, ReviewEntry

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


def _llm(key: str, *, relevant: bool = False, question: str = "first") -> PoolJudgment:
    facts = (StatedFact(fact=1, quote="q"),) if relevant else ()
    return PoolJudgment(
        question_id=question, chunk=key, model="m", verdict=Verdict(reason="r", facts=facts)
    )


def _pairs_with_false_rejections(errors: int) -> list[tuple[str, str]]:
    llm = [_llm(f"gdpr#{i}") for i in range(100)] + [_llm("gdpr#100")]  # gdpr#100: unchecked
    text = "the chunk says this"
    review = [
        _entry(f"gdpr#{i}", "relevant", 1, text).model_copy(update={"text": text})
        if i < errors
        else _entry(f"gdpr#{i}", "rejected")
        for i in range(100)
    ]
    return rejection_candidates(llm, review)


def test_the_rule_passed_all_unchecked_rejections_count() -> None:
    candidates = _pairs_with_false_rejections(4)

    assert ("first", "gdpr#100") in candidates
    assert ("first", "gdpr#3") not in candidates  # found relevant by the person
    assert len(candidates) == 97


def test_the_rule_failed_only_the_checked_rejections_count() -> None:
    candidates = _pairs_with_false_rejections(5)

    assert ("first", "gdpr#100") not in candidates
    assert len(candidates) == 95


def test_the_person_overrules_the_llm() -> None:
    text = "the chunk says this"
    llm = [_llm("gdpr#1", relevant=True), _llm("gdpr#2")]
    review = [
        _entry("gdpr#1", "rejected"),
        _entry("gdpr#2", "relevant", 1, text).model_copy(update={"text": text}),
    ]

    assert rejection_candidates(llm, review) == [("first", "gdpr#1")]


def test_an_unchecked_relevant_verdict_and_a_new_fact_are_not_rejected() -> None:
    text = "the chunk says this"
    llm = [_llm("gdpr#1", relevant=True), _llm("gdpr#2")]
    review = [_entry("gdpr#2", "new_fact", 0, text).model_copy(update={"text": text})]

    assert rejection_candidates(llm, review) == []


_WORDS = " ".join(f"word{i}" for i in range(30))


def test_the_quote_comes_from_its_chunk_as_written() -> None:
    texts = {
        "gdpr#1": "The Controller, shall\nnotify the authority of the breach at once, it says."
    }

    judgments = rejection_quotes([("q", "gdpr#1")], texts)

    assert judgments == {
        "q": (
            Quote(
                document="gdpr",
                quote="The Controller, shall notify the authority of the breach at once, it",
            ),
        )
    }


def test_the_quote_is_in_no_other_chunk() -> None:
    shared = "| Topic | Summary of the responses received | Analysis of the authorities | Changes |"
    texts = {
        "esas#1": f"{shared} first row of the table with its own words here",
        "esas#2": f"{shared} second row of the table with its own words here",
    }

    [quote] = rejection_quotes([("q", "esas#1")], texts)["q"]

    assert contains(normalize(texts["esas#1"]), quote.quote)
    assert not contains(normalize(texts["esas#2"]), quote.quote)


def test_a_short_chunk_is_quoted_whole() -> None:
    [quote] = rejection_quotes([("q", "gdpr#1")], {"gdpr#1": "Too short."})["q"]

    assert quote.quote == "Too short"


def test_a_chunk_without_its_own_words_is_an_error() -> None:
    with pytest.raises(ValueError, match="gdpr#1: no 12 words"):
        rejection_quotes([("q", "gdpr#1")], {"gdpr#1": _WORDS, "gdpr#2": _WORDS})


def test_the_rejections_are_grouped_by_question() -> None:
    texts = {"gdpr#1": "the first chunk here", "gdpr#2": "the second chunk here"}

    judgments = rejection_quotes(
        [("a", "gdpr#1"), ("a", "gdpr#2"), ("b", "gdpr#1")],
        texts,
    )

    assert [q.quote for q in judgments["a"]] == ["the first chunk here", "the second chunk here"]
    assert [q.quote for q in judgments["b"]] == ["the first chunk here"]


_HEADER = "# Judgments: a comment to keep.\n#\n# Another one.\n"
_REJECTED = Quote(document="gdpr", quote='a "quoted" passage to reject')


def test_the_rejections_are_written_after_the_comments() -> None:
    text = add_rejections(_HEADER, {"q": (_REJECTED,)})

    assert text.startswith(_HEADER)
    assert tomllib.loads(text) == {
        "q": [{"document": "gdpr", "quote": 'a "quoted" passage to reject'}]
    }


def test_the_questions_are_sorted_and_existing_rejections_kept() -> None:
    other = Quote(document="dora", quote="another passage to reject")
    once = add_rejections(_HEADER, {"z": (_REJECTED,)})

    text = add_rejections(once, {"a": (other,), "z": (other,)})

    assert list(tomllib.loads(text)) == ["a", "z"]
    assert len(tomllib.loads(text)["z"]) == 2


def test_merging_the_rejections_twice_changes_nothing() -> None:
    once = add_rejections(_HEADER, {"q": (_REJECTED,)})

    assert add_rejections(once, {"q": (_REJECTED,)}) == once


def test_a_rejected_chunk_with_a_wording_of_its_question_is_found() -> None:
    question = Question(
        id="q",
        category="factual",
        origin="blind",
        question="?",
        answer="!",
        evidence=(Evidence(any_of=(Quote(document="gdpr", quote="a wording of the fact"),)),),
    )
    other = question.model_copy(update={"id": "other", "evidence": ()})
    texts = {"gdpr#1": "It says A wording, of the fact.", "gdpr#2": "Nothing of the kind."}

    useful = stating_a_fact(
        [("q", "gdpr#1"), ("q", "gdpr#2"), ("other", "gdpr#1")],
        {"q": question, "other": other},
        texts,
    )

    assert useful == [("q", "gdpr#1")]
