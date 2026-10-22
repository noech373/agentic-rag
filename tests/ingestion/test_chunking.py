from collections import Counter
from functools import cache
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.document import Block, Chunk, Division
from agentic_rag.ingestion.chunking import chunk_blocks, count_tokens
from agentic_rag.ingestion.parsers.eurlex import parse_eurlex
from agentic_rag.ingestion.parsers.markdown import parse_markdown
from agentic_rag.ingestion.parsers.pdf import cache_path, parse_pdf

_CORPUS = Path(__file__).parents[2] / "data" / "corpus"
_CACHE = Path(__file__).parents[2] / "data" / "cache"

_SECTION = (
    Division(kind="chapter", number="IV", title="Controller and processor"),
    Division(kind="section", number="2", title="Security of personal data"),
)


def _article(number: str) -> tuple[Division, ...]:
    return (*_SECTION, Division(kind="article", number=number))


def _paragraph(article: str, number: str) -> tuple[Division, ...]:
    return (*_article(article), Division(kind="paragraph", number=number))


def _headings(*titles: str) -> tuple[Division, ...]:
    return tuple(Division(kind="heading", title=title) for title in titles)


def _text(text: str, path: tuple[Division, ...], page: int | None = None) -> Block:
    return Block(kind="text", text=text, path=path, page=page)


def _words(text: str) -> int:
    """A token counter easy to compute by hand: one token per word."""
    return len(text.split())


def _texts(chunks: list[Chunk]) -> list[str]:
    return [chunk.text for chunk in chunks]


# --- Token counting ---


def test_count_tokens_uses_the_tokenizer_of_the_embedding_model() -> None:
    assert count_tokens("hello world") == 2  # cl100k_base: "hello", " world"


def test_count_tokens_of_an_empty_text_is_zero() -> None:
    assert count_tokens("") == 0


def test_count_tokens_counts_a_special_token_as_plain_text() -> None:
    # tiktoken raises ValueError on "<|endoftext|>" by default: a document could stop the ingestion.
    assert count_tokens("<|endoftext|>") == 7


# --- The Chunk model ---


def test_a_chunk_is_frozen() -> None:
    chunk = Chunk(kind="text", text="a", path=_article("33"))

    with pytest.raises(ValidationError):
        chunk.text = "b"  # pyright: ignore[reportAttributeAccessIssue]


def test_a_chunk_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Chunk(kind="text", text="a", path=(), tokens=1)  # pyright: ignore[reportCallIssue]


def test_a_chunk_has_text() -> None:
    with pytest.raises(ValidationError):
        Chunk(kind="text", text="", path=())


def test_a_chunk_has_no_page_by_default() -> None:
    assert Chunk(kind="text", text="a", path=()).page is None


# --- One block, one chunk ---


def test_no_blocks_make_no_chunks() -> None:
    assert chunk_blocks([]) == []


def test_a_block_becomes_a_chunk_with_its_kind_text_path_and_page() -> None:
    block = Block(kind="text", text="1. In the case of a personal data breach", path=_article("33"))
    pdf_block = Block(kind="text", text="67. However", path=(), page=14)

    assert chunk_blocks([block]) == [
        Chunk(kind="text", text="1. In the case of a personal data breach", path=_article("33"))
    ]
    assert chunk_blocks([pdf_block]) == [Chunk(kind="text", text="67. However", path=(), page=14)]


def test_blocks_of_different_articles_stay_apart_in_document_order() -> None:
    blocks = [
        Block(kind="text", text="a", path=_article("33")),
        Block(kind="text", text="b", path=_article("34")),
        Block(kind="text", text="c", path=_article("35")),
    ]

    chunks = chunk_blocks(blocks)

    assert [(chunk.text, chunk.path) for chunk in chunks] == [
        ("a", _article("33")),
        ("b", _article("34")),
        ("c", _article("35")),
    ]


def test_a_table_is_a_chunk() -> None:
    table = "| Sector | Subsector |\n| --- | --- |\n| 1. Energy | (a) Electricity |"
    block = Block(kind="table", text=table, path=(Division(kind="annex", number="I"),))

    assert chunk_blocks([block]) == [Chunk(kind="table", text=table, path=block.path)]


def test_the_token_counter_and_budget_are_parameters() -> None:
    block = Block(kind="text", text="a", path=_article("33"))

    assert chunk_blocks([block], max_tokens=10, counter=_words) == chunk_blocks([block])


# --- Merging small blocks ---


def test_the_paragraphs_of_an_article_merge_into_one_chunk_with_the_article_path() -> None:
    blocks = [
        _text("1. In the case of a breach", _paragraph("33", "1")),
        _text("2. The processor shall notify", _paragraph("33", "2")),
    ]

    assert chunk_blocks(blocks, counter=_words) == [
        Chunk(
            kind="text",
            text="1. In the case of a breach\n2. The processor shall notify",
            path=_article("33"),
        )
    ]


def test_a_single_paragraph_keeps_its_own_path() -> None:
    blocks = [
        _text("1. In the case of a breach", _paragraph("33", "1")),
        _text("1. When the breach", _paragraph("34", "1")),
    ]

    assert [chunk.path for chunk in chunk_blocks(blocks, counter=_words)] == [
        _paragraph("33", "1"),
        _paragraph("34", "1"),
    ]


def test_a_merged_chunk_starts_on_the_page_of_its_first_block() -> None:
    blocks = [_text("a", _headings("A"), page=3), _text("b", _headings("A"), page=4)]

    assert chunk_blocks(blocks, counter=_words) == [
        Chunk(kind="text", text="a\nb", path=_headings("A"), page=3)
    ]


def test_merging_stops_at_the_token_budget() -> None:
    blocks = [_text(f"{n}. x", _paragraph("33", str(n))) for n in range(1, 6)]  # 2 words each

    chunks = chunk_blocks(blocks, max_tokens=4, counter=_words)

    assert _texts(chunks) == ["1. x\n2. x", "3. x\n4. x", "5. x"]


def test_the_budget_is_checked_on_the_joined_text() -> None:
    blocks = [_text("aa", _paragraph("33", "1")), _text("bb", _paragraph("33", "2"))]

    # Counting characters: "aa\nbb" is 5, one more than the two blocks.
    assert _texts(chunk_blocks(blocks, max_tokens=5, counter=len)) == ["aa\nbb"]
    assert _texts(chunk_blocks(blocks, max_tokens=4, counter=len)) == ["aa", "bb"]


def test_recitals_never_merge() -> None:
    blocks = [
        _text("(85) A breach may", (Division(kind="recital", number="85"),)),
        _text("(86) The controller", (Division(kind="recital", number="86"),)),
    ]

    assert _texts(chunk_blocks(blocks, counter=_words)) == [
        "(85) A breach may",
        "(86) The controller",
    ]


def test_text_under_different_headings_never_merges() -> None:
    blocks = [
        _text("a", _headings("II. ARTICLE 33", "A. When to notify")),
        _text("b", _headings("II. ARTICLE 33", "A. When to notify", "1. Requirements")),
        _text("c", _headings("II. ARTICLE 33", "B. Providing information")),
    ]

    assert _texts(chunk_blocks(blocks, counter=_words)) == ["a", "b", "c"]


def test_text_on_both_sides_of_an_annex_subheading_merges() -> None:
    annex = (Division(kind="annex", number="III", title="High-risk AI systems"),)
    blocks = [_text("1. Biometrics", annex), _text("2. Critical infrastructure", annex)]

    assert chunk_blocks(blocks, counter=_words) == [
        Chunk(kind="text", text="1. Biometrics\n2. Critical infrastructure", path=annex)
    ]


def test_a_point_merges_with_its_guidance_whose_headings_become_lines() -> None:
    section = _headings("3. INCIDENT HANDLING", "3.2 MONITORING AND LOGGING")
    point = (*section, Division(kind="point", number="3.2.4"))
    blocks = [
        _text("3.2.4. The logs shall be reviewed.", point),
        _text("· Consider anomaly detection.", (*point, *_headings("GUIDANCE"))),
        _text("· Regular reports.", (*point, *_headings("EXAMPLES OF EVIDENCE"))),
        _text("3.2.5. The relevant entities", (*section, Division(kind="point", number="3.2.5"))),
    ]

    chunks = chunk_blocks(blocks, counter=_words)

    assert _texts(chunks) == [
        "3.2.4. The logs shall be reviewed.\n"
        "GUIDANCE\n"
        "· Consider anomaly detection.\n"
        "EXAMPLES OF EVIDENCE\n"
        "· Regular reports.",
        "3.2.5. The relevant entities",
    ]
    assert chunks[0].path == point


def test_a_chunk_starting_under_a_heading_keeps_it_when_merged_with_the_next_one() -> None:
    point = (Division(kind="point", number="2.1.4"),)
    blocks = [
        _text("· Ensure that key personnel", (*point, *_headings("GUIDANCE"))),
        _text("· Records of", (*point, *_headings("EXAMPLES OF EVIDENCE"))),
    ]

    chunks = chunk_blocks(blocks, counter=_words)

    assert _texts(chunks) == [
        "GUIDANCE\n· Ensure that key personnel\nEXAMPLES OF EVIDENCE\n· Records of"
    ]
    assert chunks[0].path == point


def test_a_table_never_merges_with_text_or_another_table() -> None:
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    blocks = [
        _text("x", _article("33")),
        Block(kind="table", text=table, path=_article("33")),
        Block(kind="table", text=table, path=_article("33")),
        _text("y", _article("33")),
    ]

    assert [chunk.kind for chunk in chunk_blocks(blocks, counter=_words)] == [
        "text",
        "table",
        "table",
        "text",
    ]


# --- Splitting long blocks ---


def test_a_long_block_splits_between_its_lines() -> None:
    text = "Definitions apply:\n(1) controller means\n(2) processor means\n(3) recipient means"
    block = _text(text, _article("4"), page=7)  # 11 words

    chunks = chunk_blocks([block], max_tokens=6, counter=_words)

    assert chunks == [
        Chunk(
            kind="text", text="Definitions apply:\n(1) controller means", path=_article("4"), page=7
        ),
        Chunk(
            kind="text", text="(2) processor means\n(3) recipient means", path=_article("4"), page=7
        ),
    ]


def test_nested_points_stay_with_the_line_introducing_them() -> None:
    text = "(16) establishment means:\n  (a) controller\n  (b) processor\n(17) representative"

    chunks = chunk_blocks([_text(text, _article("4"))], max_tokens=7, counter=_words)

    assert _texts(chunks) == [
        "(16) establishment means:\n  (a) controller\n  (b) processor",
        "(17) representative",
    ]


def test_a_line_over_the_budget_stays_whole() -> None:
    block = _text("(53) It is also important to clarify", (Division(kind="recital", number="53"),))

    assert _texts(chunk_blocks([block], max_tokens=3, counter=_words)) == [block.text]


def test_a_table_over_the_budget_stays_whole() -> None:
    table = "| Sector | Subsector |\n| --- | --- |\n| 1. Energy | (a) Electricity |"
    block = Block(kind="table", text=table, path=(Division(kind="annex", number="I"),))

    assert chunk_blocks([block], max_tokens=3, counter=_words) == [
        Chunk(kind="table", text=table, path=block.path)
    ]


def test_the_end_of_a_long_block_merges_with_the_next_block_of_its_unit() -> None:
    blocks = [_text("a a\nb b\nc", _paragraph("4", "1")), _text("2. d", _paragraph("4", "2"))]

    chunks = chunk_blocks(blocks, max_tokens=4, counter=_words)

    assert [(chunk.text, chunk.path) for chunk in chunks] == [
        ("a a\nb b", _paragraph("4", "1")),
        ("c\n2. d", _article("4")),
    ]


# --- On the real corpus (skipped until `uv run agentic-rag download` and `convert`) ---

_ACTS = ["gdpr", "ai-act", "dora", "nis2", "data-act"]
_CNIL = [f"cnil-developer-guide-{n:02}" for n in range(17)]
_ENISA = "enisa-nis2-implementation-guidance"
_PDFS = [
    "edpb-guidelines-2022-09",
    "edpb-guidelines-2021-01",
    _ENISA,
    "esas-dora-incident-reporting",
]


def _content(filename: str) -> bytes:
    path = _CORPUS / filename
    if not path.exists():
        pytest.skip(f"{path} not found: run `uv run agentic-rag download`")
    return path.read_bytes()


@cache
def _blocks(document_id: str) -> list[Block]:
    if document_id in _ACTS:
        return parse_eurlex(_content(f"{document_id}.xhtml"))
    if document_id in _CNIL:
        return parse_markdown(_content(f"{document_id}.md"))
    content = _content(f"{document_id}.pdf")
    if not cache_path(content, _CACHE).exists():
        pytest.skip(f"{document_id} not converted: run `uv run agentic-rag convert`")
    return parse_pdf(content, cache_dir=_CACHE)


@cache
def _chunks(document_id: str) -> list[Chunk]:
    return chunk_blocks(_blocks(document_id))


def _in_article(chunk: Chunk, number: str) -> bool:
    return any(d.kind == "article" and d.number == number for d in chunk.path)


def test_corpus_gdpr_article_33_is_one_chunk() -> None:
    [chunk] = [chunk for chunk in _chunks("gdpr") if _in_article(chunk, "33")]

    assert [(d.kind, d.number) for d in chunk.path] == [
        ("chapter", "IV"),
        ("section", "2"),
        ("article", "33"),
    ]
    assert chunk.text.startswith("1. In the case of a personal data breach, the controller")
    assert "\n5. The controller shall document any personal data breaches" in chunk.text
    assert count_tokens(chunk.text) == 319


def test_corpus_ai_act_definitions_split_between_definitions() -> None:
    chunks = [chunk for chunk in _chunks("ai-act") if _in_article(chunk, "3")]

    assert len(chunks) == 8
    assert chunks[0].text.startswith(
        "For the purposes of this Regulation, the following definitions"
    )
    assert all(chunk.text.startswith("(") for chunk in chunks[1:])
    assert all(count_tokens(chunk.text) <= 512 for chunk in chunks)


def test_corpus_enisa_points_keep_their_guidance_headings() -> None:
    chunk = next(chunk for chunk in _chunks(_ENISA) if chunk.text.startswith("3.2.4. The logs"))

    assert chunk.path[-1] == Division(kind="point", number="3.2.4")
    assert "\nGUIDANCE\n· Consider the use of anomaly detection" in chunk.text
    assert "\nEXAMPLES OF EVIDENCE\n· Regular reports" in chunk.text


@pytest.mark.parametrize("document_id", [*_ACTS, *_CNIL, _ENISA])
def test_corpus_only_tables_and_single_lines_exceed_the_budget(document_id: str) -> None:
    for chunk in _chunks(document_id):
        if count_tokens(chunk.text) > 512:
            nested = chunk.text.split("\n")[1:]  # a line over the budget, with its nested lines
            assert chunk.kind == "table" or all(line.startswith(" ") for line in nested)


@pytest.mark.parametrize("document_id", [*_ACTS, *_CNIL])
def test_corpus_chunks_have_exactly_the_text_of_the_blocks(document_id: str) -> None:
    blocks, chunks = _blocks(document_id), _chunks(document_id)

    assert "\n".join(chunk.text for chunk in chunks) == "\n".join(block.text for block in blocks)


def test_corpus_enisa_chunks_only_add_the_headings_they_enter() -> None:
    blocks, chunks = _blocks(_ENISA), _chunks(_ENISA)
    headings = {d.title for block in blocks for d in block.path if d.kind == "heading"}
    added = Counter("\n".join(_texts(chunks)).split("\n")) - Counter(
        "\n".join(block.text for block in blocks).split("\n")
    )

    assert set(added) <= headings
    assert added["GUIDANCE"] > 100


def test_corpus_enisa_guidance_never_loses_its_heading() -> None:
    # Guidance and evidence are bullets; the requirement of a point is not: "3.2.4. The logs".
    for chunk in _chunks(_ENISA):
        if chunk.path and chunk.path[-1].kind == "point":
            assert not chunk.text.startswith("·"), chunk.text[:80]


def test_corpus_merging_more_than_offsets_splitting() -> None:
    documents = [*_ACTS, *_PDFS, *_CNIL]

    assert sum(len(_blocks(d)) for d in documents) == 3282
    assert sum(len(_chunks(d)) for d in documents) == 1922
