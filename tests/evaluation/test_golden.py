"""The golden set is well formed, and each of its quotes is in the corpus as it is chunked."""

from collections import Counter
from functools import cache
from pathlib import Path
from typing import NamedTuple, get_args

import pytest

from agentic_rag.evaluation.golden import (
    Category,
    Evidence,
    Question,
    Quote,
    contains,
    load_golden_set,
    normalize,
)
from agentic_rag.generation.base import NO_ANSWER
from agentic_rag.ingestion.chunking import chunk_blocks
from agentic_rag.ingestion.corpus import load_manifest
from agentic_rag.ingestion.parsers.pdf import cache_path
from agentic_rag.ingestion.pipeline import default_parsers

_ROOT = Path(__file__).parents[2]
_CORPUS = _ROOT / "data" / "corpus"
_CACHE = _ROOT / "data" / "cache"
_MANIFEST = load_manifest(_ROOT / "corpus" / "manifest.toml")
_QUESTIONS = load_golden_set(_ROOT / "eval" / "golden.toml").questions

_MIN_WORDS, _MAX_WORDS = 4, 25  # specific enough, and short enough not to straddle two chunks
_MAX_CHUNKS = 10  # a quote in more chunks is a common phrase: it would make any chunk relevant


def _each(category: str | None = None) -> list[object]:
    """One test per question (of `category`), named after its id."""
    return [
        pytest.param(question, id=question.id)
        for question in _QUESTIONS
        if category in (None, question.category)
    ]


def _quotes(question: Question) -> list[Quote]:
    return [quote for evidence in question.evidence for quote in evidence.any_of]


# --- Matching ---


def test_normalize_keeps_lowercase_words_only() -> None:
    assert normalize("Within  72 hours,\nafter (Article 33)!") == "within 72 hours after article 33"


def test_normalize_reads_check_marks_as_words() -> None:
    assert normalize("Essential ✔ | Important ✘") == "essential yes important no"


def test_contains_ignores_case_punctuation_and_spacing() -> None:
    assert contains(
        normalize("Notify it without undue delay; and, where feasible"), "undue DELAY and"
    )


def test_contains_matches_whole_words_only() -> None:
    assert not contains(normalize("within 172 hours"), "72 hours")


# --- Format ---


def test_ids_are_unique() -> None:
    counts = Counter(question.id for question in _QUESTIONS)

    assert [id_ for id_, count in counts.items() if count > 1] == []


@pytest.mark.parametrize("question", _each())
def test_only_unanswerable_questions_have_no_evidence(question: Question) -> None:
    assert (question.category == "unanswerable") == (not question.evidence)


@pytest.mark.parametrize("question", _each())
def test_only_unanswerable_questions_are_refused(question: Question) -> None:
    assert (question.category == "unanswerable") == (question.answer == NO_ANSWER)


@pytest.mark.parametrize("question", _each("multi-document"))
def test_multi_document_questions_need_two_documents(question: Question) -> None:
    titles = {document.id: document.short_title for document in _MANIFEST.documents}
    documents = {titles[evidence.any_of[0].document] for evidence in question.evidence}

    assert len(documents) >= 2  # by title: the 17 sheets of the CNIL guide are one document


@pytest.mark.parametrize("category", get_args(Category.__value__))
def test_a_fifth_of_each_category_is_held_out(category: str) -> None:
    questions = [question for question in _QUESTIONS if question.category == category]
    held_out = sum(question.holdout for question in questions)

    assert abs(held_out - len(questions) / 5) <= 1, f"{held_out} of {len(questions)} held out"


@pytest.mark.parametrize("question", _each())
def test_quotes_come_from_documents_of_the_manifest(question: Question) -> None:
    ids = {document.id for document in _MANIFEST.documents}

    assert [quote.document for quote in _quotes(question) if quote.document not in ids] == []


@pytest.mark.parametrize("question", _each())
def test_quotes_are_short(question: Question) -> None:
    for quote in _quotes(question):
        words = len(normalize(quote.quote).split())
        assert _MIN_WORDS <= words <= _MAX_WORDS, f"{quote.quote!r} has {words} words"


# --- On the real corpus (skipped until `uv run agentic-rag download` and `convert`) ---


class _Chunk(NamedTuple):
    text: str  # normalized
    table: bool


@cache
def _chunks() -> dict[str, list[_Chunk]]:
    """The chunks of each document, as the ingestion stores them."""
    paths = {document.id: _CORPUS / document.filename for document in _MANIFEST.documents}
    for document in _MANIFEST.documents:  # before parsing anything: a skip is not cached
        path = paths[document.id]
        if not path.exists():
            pytest.skip(f"{path} not found: run `uv run agentic-rag download`")
        if document.format == "pdf" and not cache_path(path.read_bytes(), _CACHE).exists():
            pytest.skip(f"{document.id} not converted: run `uv run agentic-rag convert`")
    parsers = default_parsers(_CACHE)
    return {
        document.id: [
            _Chunk(normalize(chunk.text), chunk.kind == "table")
            for chunk in chunk_blocks(parsers[document.format](paths[document.id].read_bytes()))
        ]
        for document in _MANIFEST.documents
    }


@pytest.mark.parametrize("question", _each())
def test_quotes_are_whole_in_a_chunk_of_their_document(question: Question) -> None:
    chunks = _chunks()
    for quote in _quotes(question):
        elsewhere = [
            document
            for document, document_chunks in chunks.items()
            if any(contains(chunk.text, quote.quote) for chunk in document_chunks)
        ]
        assert quote.document in elsewhere, (
            f"{quote.quote!r} is in no chunk of {quote.document}, but in: {elsewhere}"
        )


@pytest.mark.parametrize("question", _each())
def test_quotes_are_specific(question: Question) -> None:
    chunks = _chunks()
    for quote in _quotes(question):
        count = sum(
            contains(chunk.text, quote.quote)
            for document_chunks in chunks.values()
            for chunk in document_chunks
        )
        assert count <= _MAX_CHUNKS, f"{quote.quote!r} is in {count} chunks: quote more precisely"


@pytest.mark.parametrize("question", _each())
def test_quotes_are_in_no_document_of_another_regulation(question: Question) -> None:
    regulations = {document.id: set(document.regulations) for document in _MANIFEST.documents}
    chunks = _chunks()
    for quote in _quotes(question):
        foreign = [
            document
            for document, document_chunks in chunks.items()
            if not regulations[document] & regulations[quote.document]
            and any(contains(chunk.text, quote.quote) for chunk in document_chunks)
        ]
        assert foreign == [], f"{quote.quote!r} is also in {foreign}: quote more precisely"


def _only_in_tables(evidence: Evidence) -> bool:
    """Whether every chunk that states this fact is a table: prose cannot answer it."""
    found = [
        chunk
        for document_chunks in _chunks().values()
        for chunk in document_chunks
        for quote in evidence.any_of
        if contains(chunk.text, quote.quote)
    ]
    return bool(found) and all(chunk.table for chunk in found)


@pytest.mark.parametrize("question", _each("table"))
def test_table_questions_need_a_fact_stated_only_in_tables(question: Question) -> None:
    assert any(_only_in_tables(evidence) for evidence in question.evidence), (
        "every fact is also in prose: the question does not test the tables"
    )
