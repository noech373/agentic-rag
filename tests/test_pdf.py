import re
from collections import Counter
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Self

import pytest
from docling.datamodel.base_models import ConversionStatus
from docling.utils.pdf_outline import (
    _PdfOutlineItem,  # pyright: ignore[reportPrivateUsage]  # Docling has no public name for it
)
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.group import ListGroup
from docling_core.types.doc.items.picture.picture import PictureItem
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.table.table_data import TableCell, TableData
from docling_core.types.doc.items.text import ListItem, SectionHeaderItem, TextItem
from docling_core.types.doc.labels import DocItemLabel

from agentic_rag.ingestion import pdf
from agentic_rag.ingestion.document import Block, Division
from agentic_rag.ingestion.pdf import (
    PdfConversionError,
    blocks_from_docling,
    cache_path,
    load_or_convert,
    parse_pdf,
    read_outline,
)

_ROOT = Path(__file__).parents[1]
_CORPUS = _ROOT / "data" / "corpus"
_CACHE = _ROOT / "data" / "cache"


# --- Builder: Docling's output for a PDF, item by item in reading order ---


class _Pdf:
    def __init__(self) -> None:
        self.document = DoclingDocument(name="test")
        self._list: ListGroup | None = None  # consecutive list items share a list group

    def heading(self, text: str) -> Self:
        self._list = None
        self.document.add_heading(text)
        return self

    def text(self, text: str, label: DocItemLabel = DocItemLabel.TEXT, page: int = 0) -> Self:
        self._list = None
        prov = None
        if page:
            bbox = BoundingBox(l=0, t=0, r=1, b=1)
            prov = ProvenanceItem(page_no=page, bbox=bbox, charspan=(0, len(text)))
        self.document.add_text(label=label, text=text, prov=prov)
        return self

    def item(self, text: str, marker: str = "·") -> Self:
        if self._list is None:
            self._list = self.document.add_list_group()
        # Like Docling, which keeps the marker in `orig` and drops it from `text`.
        orig = f"{marker} {text}" if marker else text
        self.document.add_list_item(text, marker=marker, orig=orig, parent=self._list)
        return self

    def table(self, rows: list[list[str]], label: DocItemLabel = DocItemLabel.TABLE) -> Self:
        self._list = None
        cells = [
            TableCell(
                text=text,
                start_row_offset_idx=r,
                end_row_offset_idx=r + 1,
                start_col_offset_idx=c,
                end_col_offset_idx=c + 1,
            )
            for r, row in enumerate(rows)
            for c, text in enumerate(row)
        ]
        data = TableData(num_rows=len(rows), num_cols=len(rows[0]), table_cells=cells)
        self.document.add_table(data=data, label=label)
        return self

    def picture(self) -> Self:
        self._list = None
        self.document.add_picture()
        return self

    def blocks(self, *bookmarks: tuple[str, int]) -> list[Block]:
        """Parse, with these PDF bookmarks (title, depth from 0)."""
        outline = [_PdfOutlineItem(title=title, level=level) for title, level in bookmarks]
        return blocks_from_docling(self.document, outline)


def _label(division: Division) -> str:
    return division.title or f"{division.kind} {division.number}"


def _paths(blocks: list[Block]) -> list[str]:
    return [" > ".join(_label(division) for division in block.path) for block in blocks]


# --- Heading levels ---


def test_numbering_gives_the_heading_levels() -> None:
    blocks = (
        _Pdf()
        .heading("1 INTRODUCTION")
        .text("a")
        .heading("2 RANSOMWARE")
        .heading("2.1 CASE No. 01: Ransomware with proper backup")
        .heading("2.1.1 CASE No. 01 - Prior measures and risk assessment")
        .text("b")
        .heading("3 DATA EXFILTRATION ATTACKS")
        .text("c")
        .blocks()
    )

    assert _paths(blocks) == [
        "1 INTRODUCTION",
        "2 RANSOMWARE > 2.1 CASE No. 01: Ransomware with proper backup"
        " > 2.1.1 CASE No. 01 - Prior measures and risk assessment",
        "3 DATA EXFILTRATION ATTACKS",
    ]
    assert blocks[0].path == (Division(kind="heading", title="1 INTRODUCTION"),)


def test_bookmarks_give_the_heading_levels() -> None:
    blocks = (
        _Pdf()
        .heading("INTRODUCTION")
        .heading("Background")
        .text("a")
        .heading("Goal")
        .text("b")
        .blocks(("Introduction", 0), ("Background", 1), ("Goal", 1))
    )

    assert _paths(blocks) == ["INTRODUCTION > Background", "INTRODUCTION > Goal"]


def test_a_heading_without_bookmark_or_number_nests_under_the_numbered_ones() -> None:
    blocks = (
        _Pdf()
        .heading("1. POLICY ON THE SECURITY OF NETWORK AND INFORMATION SYSTEMS")
        .heading("1.1 POLICY ON THE SECURITY OF NETWORK AND INFORMATION SYSTEMS")
        .heading("GUIDANCE")
        .item("Establish a policy.")
        .heading("EXAMPLES OF EVIDENCE")
        .item("A documented policy.")
        .heading("1.2 ROLES, RESPONSIBILITIES AND AUTHORITIES")
        .text("a")
        .blocks()
    )

    assert [path.replace(" OF NETWORK AND INFORMATION SYSTEMS", "") for path in _paths(blocks)] == [
        "1. POLICY ON THE SECURITY > 1.1 POLICY ON THE SECURITY > GUIDANCE",
        "1. POLICY ON THE SECURITY > 1.1 POLICY ON THE SECURITY > EXAMPLES OF EVIDENCE",
        "1. POLICY ON THE SECURITY > 1.2 ROLES, RESPONSIBILITIES AND AUTHORITIES",
    ]


def test_a_list_item_promoted_to_heading_by_a_bookmark_keeps_its_number() -> None:
    # EDPB Guidelines 9/2022: "1. Article 33 requirements" is a list item for Docling.
    blocks = (
        _Pdf()
        .heading("A. When to notify")
        .item("Article 33 requirements", marker="1.")
        .text("Article 33(1) GDPR provides that:")
        .blocks(("A. When to notify", 0), ("1. Article 33 requirements", 1))
    )

    assert _paths(blocks) == ["A. When to notify > 1. Article 33 requirements"]


# --- Sentences taken for headings ---


@pytest.mark.parametrize(
    "heading", ["Whereas:", "HAS ADOPTED THIS REGULATION:", "THE EUROPEAN COMMISSION,", "of XXX"]
)
def test_a_heading_that_is_a_sentence_is_text(heading: str) -> None:
    blocks = _Pdf().heading("5. Draft implementing technical standards").heading(heading).blocks()

    assert [(block.text, _paths([block])[0]) for block in blocks] == [
        (heading, "5. Draft implementing technical standards")
    ]


def test_a_sentence_heading_does_not_shift_the_chapters_after_it() -> None:
    # EDPB Guidelines 01/2021: "49. Advisable measures:" put chapters 3 to 7 under chapter 2.
    blocks = (
        _Pdf()
        .heading("1 INTRODUCTION")
        .heading("2 RANSOMWARE")
        .heading("2.1 CASE No. 01")
        .heading("49. Advisable measures:")
        .text("a")
        .heading("3 DATA EXFILTRATION ATTACKS")
        .heading("3.1 CASE No. 05")
        .text("b")
        .blocks()
    )

    assert [(path, block.text) for path, block in zip(_paths(blocks), blocks, strict=True)] == [
        ("2 RANSOMWARE > 2.1 CASE No. 01", "49. Advisable measures:\na"),
        ("3 DATA EXFILTRATION ATTACKS > 3.1 CASE No. 05", "b"),
    ]


def test_a_question_heading_stays_a_heading() -> None:
    question = "Q1. Do you agree with the proposed timelines? If not, please explain."
    blocks = _Pdf().heading(question).table([["Comments", "Response"], ["Short.", "OK."]]).blocks()

    assert _paths(blocks) == [question]


# --- Points of a legal text (ENISA) ---


def test_a_point_is_a_division_and_its_guidance_nests_under_it() -> None:
    blocks = (
        _Pdf()
        .heading("3. INCIDENT HANDLING")
        .heading("3.5 INCIDENT RESPONSE")
        .text("3.5.1. The relevant entities shall respond to incidents in a timely manner.")
        .heading("GUIDANCE")
        .item("Establish a dedicated incident response team.")
        .heading("EXAMPLES OF EVIDENCE")
        .item("Assignment of roles within the incident response team.")
        .text("3.5.2. The incident response procedures shall include the following stages:")
        .item("(a) incident containment;", marker="")
        .heading("GUIDANCE")
        .item("Create incident response procedures.")
        .heading("3.6 POST-INCIDENT REVIEWS")
        .text("a")
        .blocks()
    )

    assert _paths(blocks) == [
        "3. INCIDENT HANDLING > 3.5 INCIDENT RESPONSE > point 3.5.1",
        "3. INCIDENT HANDLING > 3.5 INCIDENT RESPONSE > point 3.5.1 > GUIDANCE",
        "3. INCIDENT HANDLING > 3.5 INCIDENT RESPONSE > point 3.5.1 > EXAMPLES OF EVIDENCE",
        "3. INCIDENT HANDLING > 3.5 INCIDENT RESPONSE > point 3.5.2",
        "3. INCIDENT HANDLING > 3.5 INCIDENT RESPONSE > point 3.5.2 > GUIDANCE",
        "3. INCIDENT HANDLING > 3.6 POST-INCIDENT REVIEWS",
    ]
    assert blocks[3].text == (
        "3.5.2. The incident response procedures shall include the following stages:\n"
        "(a) incident containment;"
    )


def test_a_point_taken_for_a_heading_is_a_point() -> None:
    # Docling took 10 ENISA points for headings: "3.5.5. ... procedures." ends with a dot.
    blocks = (
        _Pdf()
        .heading("3.5 INCIDENT RESPONSE")
        .heading("3.5.5. The relevant entities shall test their incident response procedures.")
        .heading("GUIDANCE")
        .item("Run exercises.")
        .blocks()
    )

    assert _paths(blocks) == [
        "3.5 INCIDENT RESPONSE > point 3.5.5",
        "3.5 INCIDENT RESPONSE > point 3.5.5 > GUIDANCE",
    ]
    assert blocks[0].text.startswith("3.5.5. The relevant entities")


def test_a_numbered_heading_without_final_dot_stays_a_heading() -> None:
    heading = "6.1. Draft cost-benefit analysis / impact assessment"  # ESAs

    assert _paths(_Pdf().heading(heading).text("a").blocks()) == [heading]


# --- Blocks ---


def test_consecutive_texts_and_list_items_make_one_block_one_line_each() -> None:
    blocks = (
        _Pdf()
        .heading("GUIDANCE")
        .text("Entities should:")
        .item("take priorities into account.")
        .item("with an enumerated marker", marker="(c)")
        .item("(d) eradication;", marker="")  # Docling left the marker in the text
        .heading("TIPS")
        .text("b")
        .blocks()
    )

    assert [(block.kind, block.text) for block in blocks] == [
        (
            "text",
            "Entities should:\n"
            "· take priorities into account.\n"
            "(c) with an enumerated marker\n"
            "(d) eradication;",
        ),
        ("text", "b"),
    ]


def test_footnotes_are_skipped() -> None:
    blocks = (
        _Pdf()
        .heading("A. When to notify")
        .text("Recital 87 GDPR states 23 :")
        .text("23 Recital 85 GDPR is also important here.", label=DocItemLabel.FOOTNOTE)
        .text('"It should be ascertained whether..."')
        .blocks()
    )

    assert [block.text for block in blocks] == [
        'Recital 87 GDPR states 23 :\n"It should be ascertained whether..."'
    ]


def test_a_block_has_the_page_it_starts_on() -> None:
    blocks = (
        _Pdf()
        .heading("INTRODUCTION")
        .text("a", page=3)
        .text("b", page=4)
        .heading("SCOPE")
        .text("c", page=5)
        .blocks()
    )

    assert [(block.text, block.page) for block in blocks] == [("a\nb", 3), ("c", 5)]


def test_a_table_splits_the_text_around_it() -> None:
    blocks = _Pdf().heading("ANNEX").text("before").table([["h"], ["x"]]).text("after").blocks()

    assert [(block.kind, block.text.splitlines()[0]) for block in blocks] == [
        ("text", "before"),
        ("table", "| h |"),
        ("text", "after"),
    ]


def test_a_table_is_one_markdown_block() -> None:
    rows = [
        ["Internal documentation", "Notification to SA", "Communication to data subjects"],
        ["", "X", "a | b"],  # a Wingdings check mark
    ]
    blocks = _Pdf().heading("2.1.2 Mitigation and obligations").table(rows).blocks()

    assert blocks == [
        Block(
            kind="table",
            text=(
                "| Internal documentation | Notification to SA | Communication to data subjects |\n"
                "| --- | --- | --- |\n"
                r"| ✓ | X | a \| b |"
            ),
            path=(Division(kind="heading", title="2.1.2 Mitigation and obligations"),),
        )
    ]


def test_an_empty_table_is_skipped() -> None:
    assert _Pdf().heading("ANNEX").table([["", ""], ["", ""]]).blocks() == []


def test_pictures_and_the_table_of_contents_are_skipped() -> None:
    blocks = (
        _Pdf()
        .heading("TABLE OF CONTENTS")
        .table([["1 Introduction", "5"]], label=DocItemLabel.DOCUMENT_INDEX)
        .picture()
        .heading("1 Introduction")
        .text("a")
        .blocks()
    )

    assert _paths(blocks) == ["1 Introduction"]


def test_each_line_has_single_spaces_and_readable_bullets() -> None:
    blocks = _Pdf().heading("INTRODUCTION").text("a  personal\ndata   breach").blocks()
    bullet = _Pdf().heading("MEASURES").item("Systematic IT security audits", marker="�")

    assert blocks[0].text == "a personal data breach"
    assert bullet.blocks()[0].text == "· Systematic IT security audits"


def test_the_input_document_is_not_modified() -> None:
    builder = _Pdf().heading("Whereas:").heading("1. A").text("a")
    before = builder.document.export_to_dict()

    builder.blocks(("1. A", 0))

    assert builder.document.export_to_dict() == before


# --- Conversion and cache (Docling itself is not run) ---


def test_load_or_convert_reads_the_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    content = b"%PDF-1.7 cached"
    path = cache_path(content, tmp_path)
    path.parent.mkdir(parents=True)
    _Pdf().heading("Cached").document.save_as_json(path)

    def no_conversion(content: bytes) -> DoclingDocument:
        raise AssertionError("Docling must not run")

    monkeypatch.setattr(pdf, "convert_pdf", no_conversion)

    document = load_or_convert(content, tmp_path)

    assert [item.text for item in document.texts] == ["Cached"]


def test_load_or_convert_converts_once_and_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[bytes] = []

    def convert(content: bytes) -> DoclingDocument:
        calls.append(content)
        return _Pdf().heading("Converted").document

    monkeypatch.setattr(pdf, "convert_pdf", convert)

    load_or_convert(b"%PDF-1.7 new", tmp_path)
    document = load_or_convert(b"%PDF-1.7 new", tmp_path)

    assert calls == [b"%PDF-1.7 new"]
    assert [item.text for item in document.texts] == ["Converted"]
    assert cache_path(b"%PDF-1.7 new", tmp_path).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_the_cache_key_changes_with_the_content(tmp_path: Path) -> None:
    assert cache_path(b"a", tmp_path) != cache_path(b"b", tmp_path)
    assert cache_path(b"a", tmp_path).parent.name.startswith("docling-2.")


def test_a_partial_conversion_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Error:
        error_message = "page 3: timeout"

    class _Result:
        status = ConversionStatus.PARTIAL_SUCCESS
        errors = (_Error(),)

    class _Converter:
        def convert(self, source: object) -> _Result:
            return _Result()

    monkeypatch.setattr(pdf, "_converter", _Converter)

    with pytest.raises(PdfConversionError, match="partial_success: page 3: timeout"):
        pdf.convert_pdf(b"%PDF-1.7")


# --- On the real corpus (skipped until `uv run agentic-rag download` and `convert`) ---

_EDPB_2022 = "edpb-guidelines-2022-09"
_EDPB_2021 = "edpb-guidelines-2021-01"
_ESAS = "esas-dora-incident-reporting"
_ENISA = "enisa-nis2-implementation-guidance"
_PDFS = [_EDPB_2022, _EDPB_2021, _ESAS, _ENISA]


def _corpus(document_id: str) -> bytes:
    path = _CORPUS / f"{document_id}.pdf"
    if not path.exists():
        pytest.skip(f"{path} not found: run `uv run agentic-rag download`")
    content = path.read_bytes()
    if not cache_path(content, _CACHE).exists():
        pytest.skip(f"{document_id} not converted: run `uv run agentic-rag convert`")
    return content


@cache
def _docling(document_id: str) -> DoclingDocument:
    return load_or_convert(_corpus(document_id), _CACHE)


@cache
def _parsed(document_id: str) -> list[Block]:
    return parse_pdf(_corpus(document_id), cache_dir=_CACHE)


def _tops(document_id: str) -> set[str]:
    """The first word of each top-level heading: its number, if it has one."""
    return {(block.path[0].title or "").split()[0] for block in _parsed(document_id) if block.path}


def test_corpus_bookmarks() -> None:
    assert len(read_outline(_corpus(_EDPB_2022))) == 39
    assert read_outline(_corpus(_EDPB_2021)) == []


def test_corpus_edpb_2022_chapters_and_a_paragraph() -> None:
    block = next(b for b in _parsed(_EDPB_2022) if b.text.startswith("29. Article 33(1) GDPR"))

    assert {"I.", "II.", "III.", "IV.", "V.", "VI.", "VII."} <= _tops(_EDPB_2022)
    assert not _tops(_EDPB_2022) & {"A.", "B.", "1.", "2."}
    assert _paths([block]) == [
        "II. ARTICLE 33 - NOTIFICATION TO THE SUPERVISORY AUTHORITY"
        " > A. When to notify > 1. Article 33 requirements"
    ]
    assert block.page == 10


def test_corpus_edpb_2021_sentence_headings_do_not_shift_the_chapters() -> None:
    assert {"1", "2", "3", "4", "5", "6", "7"} <= _tops(_EDPB_2021)


def test_corpus_enisa_points_and_their_guidance() -> None:
    blocks = _parsed(_ENISA)
    points = {d.number for block in blocks for d in block.path if d.kind == "point"}
    paths = set(_paths(blocks))

    assert len(points) == 157
    assert "3. INCIDENT HANDLING > 3.5 INCIDENT RESPONSE > point 3.5.3 > GUIDANCE" in paths
    assert {"1.", "3.", "13."} <= _tops(_ENISA)


@pytest.mark.parametrize("document_id", _PDFS)
def test_corpus_only_tables_split_a_division(document_id: str) -> None:
    blocks = _parsed(document_id)

    for previous, block in pairwise(blocks):
        if previous.path == block.path:
            assert "table" in (previous.kind, block.kind)


def test_corpus_esas_tables() -> None:
    assert sum(block.kind == "table" for block in _parsed(_ESAS)) == 86


@pytest.mark.parametrize("document_id", _PDFS)
def test_corpus_blocks_have_text_a_page_and_no_unreadable_glyph(document_id: str) -> None:
    for block in _parsed(document_id):
        assert block.text.strip()
        assert block.page is not None
        assert not re.search(r"[-�]", block.text)


def _words(text: str | None) -> Counter[str]:
    return Counter(re.findall(r"\w+", (text or "").lower()))


@pytest.mark.parametrize("document_id", _PDFS)
def test_corpus_no_text_is_lost(document_id: str) -> None:
    """Every word of Docling's texts, list items and tables is in a block, footnotes aside, and
    every heading over some content is in a path (or in a block, if it was a sentence).

    Headings are checked as sets: a heading repeated as is ("GUIDANCE" again on the next page)
    gives the same path, so its content joins the block before.
    """
    expected: Counter[str] = Counter()
    headings: set[str] = set()
    items = [
        item
        for item, _ in _docling(document_id).iterate_items()
        if not isinstance(item, PictureItem)  # skipped by the parser, like the contents
        and not (isinstance(item, TableItem) and item.label == DocItemLabel.DOCUMENT_INDEX)
        and not (isinstance(item, TextItem) and item.label == DocItemLabel.FOOTNOTE)
    ]
    for item, following in zip(items, [*items[1:], None], strict=True):
        if isinstance(item, ListItem):
            expected += _words(f"{item.marker} {item.text}")
        elif isinstance(item, SectionHeaderItem):  # its number may be only in `orig`
            if following is not None and not isinstance(following, SectionHeaderItem):
                headings |= set(_words(item.orig))
        elif isinstance(item, TextItem):
            expected += _words(item.text)
        elif isinstance(item, TableItem) and item.label == DocItemLabel.TABLE:
            for row in item.data.grid:
                for cell in row:
                    expected += _words(cell.text)

    found: Counter[str] = Counter()
    in_paths: set[str] = set()
    for block in _parsed(document_id):
        found += _words(block.text)
        for division in block.path:
            in_paths |= set(_words(division.title) + _words(division.number))

    # A list item promoted to a heading by a bookmark ("1. Article 33 requirements") is in a path.
    assert expected - found - Counter(in_paths) == Counter()
    assert headings - in_paths - set(found) == set()
