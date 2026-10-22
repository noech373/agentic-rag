import re
from collections import Counter
from functools import cache
from pathlib import Path

import pytest
from lxml import etree
from lxml.etree import _Element  # pyright: ignore[reportPrivateUsage]  # no public name

from agentic_rag.document import Block, Division
from agentic_rag.ingestion.parsers.eurlex import parse_eurlex

_CORPUS = Path(__file__).parents[3] / "data" / "corpus"


# --- Builders: a minimal act in the XHTML of the Official Journal, as served by Cellar ---


def _act(body: str, doctype: str = "") -> bytes:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>{doctype}'
        f'<html xmlns="http://www.w3.org/1999/xhtml"><body>{body}</body></html>'
    ).encode()


def _enacting_terms(*content: str) -> str:
    return f'<div class="eli-subdivision" id="enc_1">{"".join(content)}</div>'


def _chapter(number: str, title: str, *content: str) -> str:
    return (
        f'<div id="cpt_{number}"><p class="oj-ti-section-1">CHAPTER {number}</p>'
        f'<div class="eli-title" id="cpt_{number}.tit_1"><p class="oj-ti-section-2">{title}</p>'
        f"</div>{''.join(content)}</div>"
    )


def _article(number: int, title: str, *content: str) -> str:
    return (
        f'<div class="eli-subdivision" id="art_{number}">'
        f'<p class="oj-ti-art">Article {number}</p>'
        f'<div class="eli-title" id="art_{number}.tit_1"><p class="oj-sti-art">{title}</p></div>'
        f"{''.join(content)}</div>"
    )


def _paragraph(article: int, number: int, *content: str) -> str:
    return f'<div id="{article:03}.{number:03}">{"".join(content)}</div>'


def _p(text: str, css_class: str = "oj-normal") -> str:
    return f'<p class="{css_class}">{text}</p>'


def _point(label: str, *body: str) -> str:
    """A point (a), (b)... laid out as a one-row table, like in the Official Journal."""
    return (
        '<table width="100%" border="0" cellspacing="0" cellpadding="0">'
        '<col width="4%"/><col width="96%"/><tbody><tr>'
        f'<td valign="top"><p class="oj-normal">{label}</p></td>'
        f'<td valign="top">{"".join(body)}</td>'
        "</tr></tbody></table>"
    )


def _data_table(*rows: str) -> str:
    return f'<table class="oj-table"><tbody>{"".join(rows)}</tbody></table>'


def _row(*cells: str) -> str:
    return "<tr>" + "".join(cells) + "</tr>"


def _cell(text: str, rowspan: int = 1, colspan: int = 1) -> str:
    return f'<td rowspan="{rowspan}" colspan="{colspan}"><p class="oj-tbl-txt">{text}</p></td>'


def _gdpr_article_33() -> bytes:
    return _act(
        _enacting_terms(
            '<div id="cpt_IV"><p class="oj-ti-section-1">CHAPTER IV</p>'
            '<div class="eli-title" id="cpt_IV.tit_1"><p>Controller and processor</p></div>'
            '<div id="cpt_IV.sct_2"><p class="oj-ti-section-1">Section 2</p>'
            '<div class="eli-title" id="cpt_IV.sct_2.tit_1"><p>Security of personal data</p></div>'
            + _article(
                33,
                "Notification of a personal data breach to the supervisory authority",
                _paragraph(33, 1, _p("1.\u00a0\u00a0\u00a0In the case of a personal data breach")),
                _paragraph(
                    33,
                    3,
                    _p("3.\u00a0\u00a0\u00a0The notification referred to in paragraph 1 shall:"),
                    _point("(a)", _p("describe the nature of the personal data breach;")),
                    _point("(b)", _p("communicate the name of the data protection officer;")),
                ),
            )
            + "</div></div>"
        )
    )


def _paths(blocks: list[Block]) -> list[str]:
    return [" > ".join(f"{d.kind} {d.number}" for d in block.path) for block in blocks]


# --- Structure ---


def test_each_paragraph_is_a_block_with_its_full_path() -> None:
    blocks = parse_eurlex(_gdpr_article_33())

    assert _paths(blocks) == [
        "chapter IV > section 2 > article 33 > paragraph 1",
        "chapter IV > section 2 > article 33 > paragraph 3",
    ]
    assert blocks[0].path == (
        Division(kind="chapter", number="IV", title="Controller and processor"),
        Division(kind="section", number="2", title="Security of personal data"),
        Division(
            kind="article",
            number="33",
            title="Notification of a personal data breach to the supervisory authority",
        ),
        Division(kind="paragraph", number="1"),
    )


def test_headings_are_in_the_path_not_in_the_text() -> None:
    blocks = parse_eurlex(_gdpr_article_33())

    assert blocks[0].text == "1. In the case of a personal data breach"


def test_points_are_lines_starting_with_their_label() -> None:
    blocks = parse_eurlex(_gdpr_article_33())

    assert blocks[1].text == (
        "3. The notification referred to in paragraph 1 shall:\n"
        "(a) describe the nature of the personal data breach;\n"
        "(b) communicate the name of the data protection officer;"
    )


def test_nested_points_and_their_subparagraphs_are_indented() -> None:
    article = _article(
        3,
        "Definitions",
        _p("For the purposes of this Regulation:"),
        _point(
            "(1)",
            _p("'AI system' means a system that:"),
            _point("(a)", _p("is machine-based;"), _p("This excludes simple software.")),
            _point("(b)", _p("infers from the input it receives;")),
        ),
        _point("(2)", _p("'risk' means the combination of probability and severity;")),
    )

    (block,) = parse_eurlex(_act(_enacting_terms(article)))

    assert _paths([block]) == ["article 3"]  # no paragraph: the article is a single block
    assert block.text == (
        "For the purposes of this Regulation:\n"
        "(1) 'AI system' means a system that:\n"
        "  (a) is machine-based;\n"
        "    This excludes simple software.\n"
        "  (b) infers from the input it receives;\n"
        "(2) 'risk' means the combination of probability and severity;"
    )


def test_point_with_an_indentation_column() -> None:
    # Annex I of the AI Act: an empty first cell, the label, then the text in a <span>.
    point = (
        '<table><col width="4%"/><col width="4%"/><col width="92%"/><tbody><tr>'
        '<td valign="top"/><td valign="top"><p class="oj-normal">1.</p></td>'
        '<td valign="top"><span>Directive 2006/42/EC on machinery;</span></td>'
        "</tr></tbody></table>"
    )

    (block,) = parse_eurlex(_act(_enacting_terms(_article(1, "Scope", point))))

    assert block.text == "1. Directive 2006/42/EC on machinery;"


def test_numbered_point_split_over_two_elements_is_one_line() -> None:
    point = '<div class="oj-enumeration-spacing"><p>1.</p><p>The provider verifies.</p></div>'

    (block,) = parse_eurlex(_act(_enacting_terms(_article(1, "Scope", point))))

    assert block.text == "1. The provider verifies."


def test_quoted_text_stays_in_its_paragraph() -> None:
    # DORA Article 62 quotes paragraphs of another regulation, with ids like "027H.005": they
    # are content of the amending paragraph, not paragraphs of DORA.
    paragraph = _paragraph(
        62,
        2,
        _p("(2) Article 27h is amended as follows:"),
        '<div id="027H.005"><p class="oj-normal">"5. A CTP shall comply with DORA.".</p></div>',
    )

    blocks = parse_eurlex(_act(_enacting_terms(_article(62, "Amendments", paragraph))))

    assert _paths(blocks) == ["article 62 > paragraph 2"]
    assert blocks[0].text == (
        '(2) Article 27h is amended as follows:\n"5. A CTP shall comply with DORA.".'
    )


def test_recitals() -> None:
    recital = (
        '<div class="eli-subdivision" id="pbl_1">'
        + _p("THE EUROPEAN PARLIAMENT AND THE COUNCIL OF THE EUROPEAN UNION,")
        + '<div class="eli-subdivision" id="cit_1">'
        + _p("Having regard to the Treaty on the Functioning of the European Union,")
        + "</div>"
        + _p("Whereas:")
        + '<div class="eli-subdivision" id="rct_85">'
        + _point("(85)", _p("A personal data breach may result in damage."))
        + "</div></div>"
    )

    blocks = parse_eurlex(_act(recital))

    assert blocks == [
        Block(
            kind="text",
            text="(85) A personal data breach may result in damage.",
            path=(Division(kind="recital", number="85"),),
        )
    ]


def test_title_and_final_provisions_are_skipped() -> None:
    body = (
        '<div class="eli-main-title" id="tit_1"><p class="oj-doc-ti">REGULATION (EU) 2016/679</p>'
        "</div>"
        + _enacting_terms(_article(99, "Entry into force", _p("This Regulation shall apply.")))
        + '<div class="eli-subdivision" id="fnp_1"><div class="oj-final">'
        + _p("This Regulation shall be binding in its entirety.")
        + '<div class="oj-signatory">The President</div></div></div>'
    )

    blocks = parse_eurlex(_act(body))

    assert [block.text for block in blocks] == ["This Regulation shall apply."]


def test_articles_and_chapters_keep_document_order() -> None:
    body = _enacting_terms(
        _chapter("I", "General provisions", _article(1, "Subject matter", _p("One."))),
        _chapter("II", "Principles", _article(2, "Lawfulness", _p("Two."))),
    )

    blocks = parse_eurlex(_act(body))

    assert _paths(blocks) == ["chapter I > article 1", "chapter II > article 2"]
    assert [block.text for block in blocks] == ["One.", "Two."]


# --- Annexes and data tables ---


def _annex(number: str, title: str, *content: str) -> str:
    return (
        f'<div class="eli-container" id="anx_{number}">'
        f'<p class="oj-doc-ti">ANNEX {number}</p><p class="oj-doc-ti">{title}</p>'
        f"{''.join(content)}</div>"
    )


def test_annex_title_and_subheadings() -> None:
    annex = _annex(
        "I",
        "List of Union harmonisation legislation",
        _p("Section A. Legislation based on the New Legislative Framework", "oj-ti-grseq-1"),
        _point("1.", _p("Directive 2006/42/EC on machinery;")),
        _p("Section B. Other Union harmonisation legislation", "oj-ti-grseq-1"),
        _point("13.", _p("Regulation (EC) No 300/2008 on civil aviation security;")),
    )

    blocks = parse_eurlex(_act(annex))

    # Each subheading starts a new block, which the chunker will never merge across divisions.
    assert [block.path for block in blocks] == [
        (Division(kind="annex", number="I", title="List of Union harmonisation legislation"),),
    ] * 2
    assert [block.text for block in blocks] == [
        "Section A. Legislation based on the New Legislative Framework\n"
        "1. Directive 2006/42/EC on machinery;",
        "Section B. Other Union harmonisation legislation\n"
        "13. Regulation (EC) No 300/2008 on civil aviation security;",
    ]


def test_data_table_is_its_own_markdown_block() -> None:
    annex = _annex(
        "III",
        "Correlation table",
        _p("The correlation table:"),
        _data_table(
            _row(_cell("Directive (EU) 2016/1148"), _cell("This Directive")),
            _row(_cell("Article 1(1)"), _cell("Article 1(1)")),
        ),
        _p("End of the annex."),
    )

    blocks = parse_eurlex(_act(annex))

    assert [(block.kind, block.text) for block in blocks] == [
        ("text", "The correlation table:"),
        (
            "table",
            "| Directive (EU) 2016/1148 | This Directive |\n"
            "| --- | --- |\n"
            "| Article 1(1) | Article 1(1) |",
        ),
        ("text", "End of the annex."),
    ]


def test_merged_cells_are_repeated_in_every_row_and_column() -> None:
    # Annex I of NIS2: "Energy" spans the rows of all its subsectors.
    table = _data_table(
        _row(_cell("Sector"), _cell("Subsector"), _cell("Type of entity")),
        _row(
            _cell("1. Energy", rowspan=3), _cell("(a) Electricity", rowspan=2), _cell("Producers")
        ),
        _row(_cell("Transmission system operators")),
        _row(_cell("(b) Oil"), _cell("Pipeline operators")),
        _row(_cell("2. Transport", colspan=2), _cell("Airports")),
    )

    (block,) = parse_eurlex(_act(_annex("I", "Sectors of high criticality", table)))

    assert block.text == (
        "| Sector | Subsector | Type of entity |\n"
        "| --- | --- | --- |\n"
        "| 1. Energy | (a) Electricity | Producers |\n"
        "| 1. Energy | (a) Electricity | Transmission system operators |\n"
        "| 1. Energy | (b) Oil | Pipeline operators |\n"
        "| 2. Transport | 2. Transport | Airports |"
    )


def test_table_cells_are_single_line_and_escape_pipes() -> None:
    table = _data_table(
        _row(_cell("Sector"), _cell("Type of entity")),
        _row(
            _cell("Energy"),
            "<td>" + _point("—", _p("Producers")) + _point("—", _p("Operators a|b")) + "</td>",
        ),
    )

    (block,) = parse_eurlex(_act(_annex("I", "Sectors", table)))

    assert block.text.splitlines()[-1] == r"| Energy | — Producers — Operators a\|b |"


def test_data_table_inside_a_point_is_rejected() -> None:
    point = _point("(a)", _data_table(_row(_cell("x"))))

    with pytest.raises(ValueError, match="data table"):
        parse_eurlex(_act(_enacting_terms(_article(1, "Scope", point))))


# --- Inline content ---


def test_footnotes_are_dropped_and_the_text_around_them_is_kept() -> None:
    call = (
        '\u00a0<a id="ntc1-L_2016119EN.01000101-E0001" href="#ntr1-L_2016119EN.01000101-E0001">'
        '(<span class="oj-super oj-note-tag">1</span>)</a>'
    )
    body = _enacting_terms(
        _article(
            1,
            "Scope",
            _p(f"Having regard to the opinion of the Committee{call}, and after consultation."),
            _p(f"See Directive 95/46/EC{call} for details{call}."),
            _p("(1) OJ C 229, 31.7.2012, p. 90.", "oj-note"),
        )
    )

    (block,) = parse_eurlex(_act(body))

    assert block.text == (
        "Having regard to the opinion of the Committee, and after consultation.\n"
        "See Directive 95/46/EC for details."
    )


def test_superscripts_are_written_as_exponents() -> None:
    text = 'greater than 10<span class="oj-super">25</span> floating point operations.'

    (block,) = parse_eurlex(_act(_enacting_terms(_article(51, "Classification", _p(text)))))

    assert block.text == "greater than 10^25 floating point operations."


def test_external_entities_are_not_resolved(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET")
    doctype = f'<!DOCTYPE html [<!ENTITY xxe SYSTEM "{secret.as_uri()}">]>'
    body = _enacting_terms(_article(1, "Scope", _p("Before &xxe; after.")))

    blocks = parse_eurlex(_act(body, doctype))

    assert "TOP-SECRET" not in "".join(block.text for block in blocks)


# --- On the real corpus (skipped until `uv run agentic-rag download` has been run) ---


_ACTS = ["gdpr", "ai-act", "dora", "nis2", "data-act"]


def _corpus(document_id: str) -> bytes:
    path = _CORPUS / f"{document_id}.xhtml"
    if not path.exists():
        pytest.skip(f"{path} not found: run `uv run agentic-rag download`")
    return path.read_bytes()


@cache
def _parse_corpus(document_id: str) -> list[Block]:
    return parse_eurlex(_corpus(document_id))


def _numbers(blocks: list[Block], kind: str) -> list[str | None]:
    """The numbers of the divisions of a kind, in document order, without repetition."""
    numbers = [d.number for block in blocks for d in block.path if d.kind == kind]
    return list(dict.fromkeys(numbers))


@pytest.mark.parametrize(
    ("document_id", "articles", "recitals", "annexes"),
    [
        ("gdpr", 99, 173, 0),
        ("ai-act", 113, 180, 13),
        ("dora", 64, 106, 0),
        ("nis2", 46, 144, 3),
        ("data-act", 50, 119, 0),
    ],
)
def test_corpus_acts_are_complete(
    document_id: str, articles: int, recitals: int, annexes: int
) -> None:
    blocks = _parse_corpus(document_id)

    assert _numbers(blocks, "article") == [str(n) for n in range(1, articles + 1)]
    assert _numbers(blocks, "recital") == [str(n) for n in range(1, recitals + 1)]
    assert len(_numbers(blocks, "annex")) == annexes
    assert all(block.path for block in blocks)
    # Layout tables are text: only `oj-table` tables (NIS2 annexes) become table blocks.
    assert all("| --- |" not in block.text for block in blocks if block.kind == "text")


def _words(text: str | None) -> Counter[str]:
    return Counter(re.findall(r"\w+", text or ""))


def _kept_words(element: _Element) -> Counter[str]:
    """The words the parser must keep, counted much more simply than the parser finds them:
    all the text, minus headings and titles (they are in the path) and footnotes."""
    css = element.get("class", "")
    if css in {"oj-ti-art", "oj-ti-section-1", "oj-doc-ti", "oj-note", "eli-title"}:
        return Counter()
    if element.get("id", "").startswith("ntc"):  # a footnote call; its tail is kept
        return Counter()
    words = _words(element.text)
    for child in element.iterchildren(etree.Element):
        words.update(_kept_words(child))
        words.update(_words(child.tail))
    return words


@pytest.mark.parametrize("document_id", _ACTS)
def test_corpus_no_text_is_lost_or_invented(document_id: str) -> None:
    # This check found "10^25" (the AI Act threshold for systemic risk) read as "1025".
    root = etree.fromstring(_corpus(document_id), etree.XMLParser(resolve_entities=False))
    expected: Counter[str] = Counter()
    for div in root.iter("{http://www.w3.org/1999/xhtml}div"):
        id_ = div.get("id", "")
        if id_ == "enc_1" or id_.startswith(("rct_", "anx_")):
            expected.update(_kept_words(div))
    blocks = _parse_corpus(document_id)
    found = {"text": Counter[str](), "table": Counter[str]()}
    for block in blocks:
        found[block.kind].update(_words(block.text))

    assert expected - found["text"] - found["table"] == Counter()  # nothing lost
    assert found["text"] - expected == Counter()  # nothing invented (tables repeat merged cells)


def test_corpus_gdpr_article_33_1() -> None:
    blocks = _parse_corpus("gdpr")

    (block,) = [b for b in blocks if _paths([b])[0].endswith("article 33 > paragraph 1")]
    assert "not later than 72 hours after having become aware of it" in block.text
    assert block.path[2].title == (
        "Notification of a personal data breach to the supervisory authority"
    )


def test_corpus_ai_act_systemic_risk_threshold() -> None:
    blocks = _parse_corpus("ai-act")

    (block,) = [b for b in blocks if _paths([b])[0].endswith("article 51 > paragraph 2")]
    assert "greater than 10^25." in block.text


def test_corpus_nis2_annex_tables() -> None:
    tables = [block for block in _parse_corpus("nis2") if block.kind == "table"]

    assert [table.path[0].number for table in tables] == ["I", "II", "III"]
    header, separator, *rows = tables[0].text.splitlines()
    assert header == "| Sector | Subsector | Type of entity |"
    assert separator == "| --- | --- | --- |"
    assert all(row.startswith("| ") and row.count(" | ") == 2 for row in rows)
    assert "| 1. Energy | (a) Electricity | — Producers as defined in" in tables[0].text
