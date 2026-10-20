"""Parser for EU legal acts in the XHTML of the Official Journal (as served by Cellar).

The ELI identifiers in the XHTML give the exact structure of the act: `cpt_IV` (Chapter IV),
`cpt_IV.sct_2` (its Section 2), `art_33` (Article 33), `033.001` (its paragraph 1), `rct_85`
(recital 85), `anx_III` (Annex III). Points (a), (b)... are laid out with one-row tables, which
a generic HTML converter takes for data tables (Docling finds 563 of them in the GDPR and no
heading at all); here they become lines of text. Only `oj-table` tables are data tables.

Kept: recitals, enacting terms (chapters, sections, articles, paragraphs) and annexes.
Skipped: the title (already in the manifest), citations ("Having regard to..."), final
formulas, signatures and footnotes (references to Official Journal issues).
"""

import re
from collections.abc import Iterator

from lxml import etree
from lxml.etree import _Element  # pyright: ignore[reportPrivateUsage]  # no public name

from agentic_rag.ingestion.document import Block, Division, DivisionKind

_X = "{http://www.w3.org/1999/xhtml}"  # every tag is namespaced: f"{_X}p" matches <p>
_INDENT = "  "  # per level of nested points

# Never resolve entities nor load the DTD named in the DOCTYPE: safe against XXE attacks.
_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)

_DIVISION_KINDS: dict[str, DivisionKind] = {
    "cpt": "chapter",
    "sct": "section",
    "art": "article",
    "rct": "recital",
    "anx": "annex",
}
_DIVISION_ID = re.compile(r"(?:^|\.)(?P<prefix>[a-z]{3})_(?P<number>[^.]+)$")  # last segment
_PARAGRAPH_ID = re.compile(r"^\d{3}\.(?P<number>\d{3})$")  # "033.001": Article 33(1)

# CSS classes of the Official Journal XHTML.
_TITLE = "eli-title"  # <div class="eli-title" id="art_33.tit_1">: the title of a division
_ANNEX_TITLE = "oj-doc-ti"  # <p class="oj-doc-ti">ANNEX III</p><p class="oj-doc-ti">title</p>
_HEADINGS = frozenset({"oj-ti-art", "oj-ti-section-1", _ANNEX_TITLE})  # "Article 33": in the path
_SUBHEADING = "oj-ti-grseq-1"  # "Section A. ..." inside an annex: starts a new block
_NOTE = "oj-note"
_DATA_TABLE = "oj-table"
_SPLIT_POINT = "oj-enumeration-spacing"  # <p>1.</p><p>text</p>: the label and text of a point
_SUPERSCRIPT = "oj-super"


def parse_eurlex(content: bytes) -> list[Block]:
    """Split an EU legal act into blocks, each one tagged with its path in the act."""
    root = etree.fromstring(content, _PARSER)
    _drop_note_calls(root)
    _mark_exponents(root)
    walker = _Walker()
    for div in root.iter(f"{_X}div"):  # in document order: recitals, enacting terms, annexes
        id_ = div.get("id", "")
        if id_.startswith(("rct_", "anx_")):
            walker.visit(div)
        elif id_ == "enc_1":  # the enacting terms: chapters, or articles if there is no chapter
            walker.visit_children(div)
    return walker.blocks


class _Walker:
    """Walks the tree depth-first, keeps track of the current path and collects the blocks."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self._path: tuple[Division, ...] = ()
        self._lines: list[str] = []  # the text block being built

    def visit(self, element: _Element) -> None:
        """Handle one element of the act, depending on what it is:

        - a division (a div that `_division` recognises): its content goes under a longer path;
        - a division title (`_TITLE`): nothing to do, `_division` has already read it;
        - a data table (`_DATA_TABLE`): a "table" block of its own (`_markdown_table`);
        - anything else is running text (`_lines`); an annex subheading (`_SUBHEADING`)
          starts a new block.
        """
        division = _division(element) if element.tag == f"{_X}div" else None
        if division is not None:
            self._flush()
            parent = self._path
            self._path = (*parent, division)
            self.visit_children(element)
            self._flush()
            self._path = parent
        elif _has_class(element, _TITLE):
            return
        elif element.tag == f"{_X}table" and _has_class(element, _DATA_TABLE):
            self._flush()
            self.blocks.append(Block(kind="table", text=_markdown_table(element), path=self._path))
        else:
            if _has_class(element, _SUBHEADING):
                self._flush()
            self._lines.extend(_lines(element, depth=0))

    def visit_children(self, element: _Element) -> None:
        for child in _children(element):
            self.visit(child)

    def _flush(self) -> None:
        """Turn the lines collected so far into a text block."""
        if self._lines:
            text = "\n".join(self._lines)
            self.blocks.append(Block(kind="text", text=text, path=self._path))
            self._lines = []


def _division(div: _Element) -> Division | None:
    """The Division a div stands for, recognised by its ELI id; None for any other div.

    "033.001" -> paragraph "1"; "cpt_IV.sct_2" -> section "2", with its title (`_title`).
    None for "art_33.tit_1" (a title), "enc_1", "027H.005" (quoted text), or a div without id.
    """
    id_ = div.get("id")
    if id_ is None:
        return None
    if match := _PARAGRAPH_ID.match(id_):
        return Division(kind="paragraph", number=str(int(match["number"])))
    match = _DIVISION_ID.search(id_)
    if match is None or match["prefix"] not in _DIVISION_KINDS:
        return None
    kind = _DIVISION_KINDS[match["prefix"]]
    return Division(kind=kind, number=match["number"], title=_title(div, kind))


def _title(div: _Element, kind: DivisionKind) -> str | None:
    """The title in the `eli-title` div of a division; for an annex, its second heading line."""
    if kind == "annex":
        lines = [_text(p) for p in _children(div) if _has_class(p, _ANNEX_TITLE)]
        return " ".join(lines[1:]) or None
    for child in _children(div):
        if _has_class(child, _TITLE):
            return _text(child) or None
    return None


def _lines(element: _Element, depth: int) -> list[str]:
    """Render one element of running text as lines: one per subparagraph or point.

    - a table is a point (`_point_lines`), unless it is a data table: ValueError, as a data
      table inside running text would end up in a text block;
    - a `_SPLIT_POINT` div is one line: "1." and its text;
    - any other div (e.g. quoted text) gives the lines of its children;
    - a heading (`_HEADINGS`, already in the path) or a footnote (`_NOTE`) gives nothing;
    - anything else (p, span) is one line, indented by `depth` levels, unless it is empty.
    """
    if element.tag == f"{_X}table":
        if _has_class(element, _DATA_TABLE):
            raise ValueError("unexpected data table inside running text")
        return _point_lines(element, depth)
    if element.tag == f"{_X}div":
        if _has_class(element, _SPLIT_POINT):
            return [_INDENT * depth + " ".join(_text(child) for child in _children(element))]
        return [line for child in _children(element) for line in _lines(child, depth)]
    if _has_class(element, _NOTE) or element.get("class") in _HEADINGS:
        return []
    text = _text(element)
    return [_INDENT * depth + text] if text else []


def _point_lines(table: _Element, depth: int) -> list[str]:
    """A point laid out as a one-row table: the label "(a)" first, the text in the last cell.

    The first line gets the label, e.g. "(a) describe the nature of the breach;". What follows
    (another subparagraph, nested points (i), (ii)...) is indented one level deeper.
    Annex I of the AI Act adds an empty first cell: the label is in the cells before the last.
    """
    (row,) = _rows(table)  # unpacking also checks that there is exactly one row
    *label_cells, body = row.iterchildren(f"{_X}td")
    label = " ".join(text for cell in label_cells if (text := _text(cell)))
    lines = [line for child in _children(body) for line in _lines(child, depth + 1)]
    if not lines:
        return [_INDENT * depth + label] if label else []
    first = " ".join(part for part in (label, lines[0].lstrip()) if part)
    return [_INDENT * depth + first, *lines[1:]]


def _markdown_table(table: _Element) -> str:
    """Render a data table as Markdown, its first row as the header.

    A merged cell (rowspan, colspan) is repeated in every row and column it covers, so that
    each row still makes sense on its own once the chunker splits a long table.
    """
    grid: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}  # (row, column) -> text of a cell merged from above
    for r, tr in enumerate(_rows(table)):
        row: list[str] = []
        for td in tr.iterchildren(f"{_X}td", f"{_X}th"):
            while (r, len(row)) in pending:
                row.append(pending.pop((r, len(row))))
            text = _cell_text(td)
            rowspan, colspan = int(td.get("rowspan", "1")), int(td.get("colspan", "1"))
            for column in range(len(row), len(row) + colspan):
                for below in range(r + 1, r + rowspan):
                    pending[below, column] = text
            row.extend([text] * colspan)
        while (r, len(row)) in pending:
            row.append(pending.pop((r, len(row))))
        grid.append(row)

    width = max(len(row) for row in grid)
    lines = [_markdown_row([*row, *[""] * (width - len(row))]) for row in grid]
    lines.insert(1, _markdown_row(["---"] * width))
    return "\n".join(lines)


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _cell_text(td: _Element) -> str:
    """A table cell on one line (Markdown has no line breaks in cells), `|` escaped."""
    lines = [line.strip() for child in _children(td) for line in _lines(child, depth=0)]
    return " ".join(lines).replace("|", r"\|")


def _drop_note_calls(root: _Element) -> None:
    """Remove the footnote calls, e.g. the "(1)" in "the Committee (1), ...".

    In lxml the text after an element is its `tail`, and removing the element removes its tail
    too: move the tail to the previous sibling or to the parent first. The no-break space
    before the call goes too, so that "Committee (1)," becomes "Committee,".
    """
    calls = [a for a in root.iter(f"{_X}a") if a.get("id", "").startswith("ntc")]
    for call in calls:
        parent, previous = call.getparent(), call.getprevious()
        if parent is None:
            continue
        tail = call.tail or ""
        if previous is None:
            parent.text = (parent.text or "").rstrip() + tail
        else:
            previous.tail = (previous.tail or "").rstrip() + tail
        parent.remove(call)


def _mark_exponents(root: _Element) -> None:
    """Write superscripts as exponents: "10<sup>25</sup> operations" must not read "1025"."""
    for span in root.iter(f"{_X}span"):
        if _has_class(span, _SUPERSCRIPT):
            span.text = "^" + (span.text or "")


def _rows(table: _Element) -> list[_Element]:
    return [*table.iterfind(f"{_X}tbody/{_X}tr"), *table.iterfind(f"{_X}tr")]


def _children(element: _Element) -> Iterator[_Element]:
    """Child elements, without comments, processing instructions or entities."""
    return element.iterchildren(etree.Element)


def _has_class(element: _Element, name: str) -> bool:
    return name in element.get("class", "").split()


def _text(element: _Element) -> str:
    """All the text inside an element, with whitespace (including no-break spaces) collapsed."""
    return " ".join("".join(element.itertext()).split())
