"""Parser for PDF guidance documents (EDPB, ENISA, ESAs), built on Docling.

Docling's layout model (with TableFormer for tables) turns a PDF into headings, text, list
items and tables in reading order. No OCR: every PDF of the corpus has a text layer, which is
exact, while OCR could only add errors. Docling's heading stage then nests the headings, from
the PDF bookmarks and the numbering ("I." > "A." > "1."; "2" > "2.1" > "2.1.1").

The points of a legal text quoted by a guidance document ("3.5.1. The relevant entities
shall...", from the Annex to Implementing Regulation 2024/2690 in ENISA's) are divisions too,
with the guidance on them nested below. Consecutive texts and list items under the same
divisions make one block, one line each; a table is a block of its own. Each block has the page
it starts on: what answers cite ("EDPB Guidelines 9/2022, p. 10"). Footnotes are skipped, as by
the EUR-Lex parser.

Converting is the slow part (minutes and 4.5 GB of RAM for 170 pages on CPU), so `parse_pdf`
can cache Docling's output; the rest takes about a second.
"""

import re
from collections.abc import Sequence
from functools import cache
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium  # pyright: ignore[reportMissingTypeStubs]
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.stages.heading_hierarchy.heading_hierarchy_model import HeadingHierarchyModel
from docling.utils.pdf_outline import (
    _PdfOutlineItem,  # pyright: ignore[reportPrivateUsage]  # Docling has no public name for it
    extract_outline_from_pdfium,
)
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.text import ListItem, SectionHeaderItem, TextItem
from docling_core.types.doc.labels import DocItemLabel

from agentic_rag.ingestion.document import Block, Division

type Outline = Sequence[_PdfOutlineItem]  # the PDF bookmarks, flat, in document order

_POINT_LEVEL = 99  # below the headings the stage numbers (1 to 6), above the unnumbered ones
_UNNUMBERED = 100  # the level the heading stage leaves to headings it has no signal for
_POINT = re.compile(r"(?P<number>\d{1,2}(?:\.\d{1,2})+)\.\s")  # "3.5.1. ", "7.1. "
# Wingdings glyphs, and the bullets of EDPB Guidelines 01/2021 that pdfium cannot decode.
_GLYPHS = str.maketrans({"": "✓", "": "▪", "": "➢", "�": "·"})


class PdfConversionError(Exception):
    """Docling could not convert the whole PDF."""


@cache
def _converter() -> DocumentConverter:
    """Docling's PDF pipeline, built once: loading the layout and table models takes seconds."""
    options = PdfPipelineOptions(do_ocr=False)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def convert_pdf(content: bytes) -> DoclingDocument:
    """Run Docling's layout and table models on a PDF (downloads ~500 MB of models once)."""
    result = _converter().convert(DocumentStream(name="document.pdf", stream=BytesIO(content)))
    if result.status is not ConversionStatus.SUCCESS:  # PARTIAL_SUCCESS would lose pages
        errors = "; ".join(error.error_message for error in result.errors)
        raise PdfConversionError(f"{result.status.value}: {errors or 'no details'}")
    return result.document


def cache_path(content: bytes, cache_dir: Path) -> Path:
    """Where Docling's output for this PDF is cached: by file hash and Docling version.

    Change the conversion options in `_converter` and the cache must be deleted by hand.
    """
    return cache_dir / f"docling-{version('docling-slim')}" / f"{sha256(content).hexdigest()}.json"


def load_or_convert(content: bytes, cache_dir: Path) -> DoclingDocument:
    """Docling's output for a PDF, read from the cache or converted once and cached."""
    path = cache_path(content, cache_dir)
    if path.exists():
        return DoclingDocument.load_from_json(path)
    document = convert_pdf(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    document.save_as_json(partial)
    partial.replace(path)  # atomic: an interrupted run never leaves a truncated cache entry
    return document


def read_outline(content: bytes) -> list[_PdfOutlineItem]:
    """The PDF bookmarks (outline), flat and in document order; empty if the PDF has none."""
    pdf = pdfium.PdfDocument(content)
    try:
        return extract_outline_from_pdfium(pdf)
    finally:
        pdf.close()


def parse_pdf(content: bytes, *, cache_dir: Path | None = None) -> list[Block]:
    """Split a PDF guidance document into blocks, each with its divisions and page."""
    document = convert_pdf(content) if cache_dir is None else load_or_convert(content, cache_dir)
    return blocks_from_docling(document, read_outline(content))


def blocks_from_docling(document: DoclingDocument, outline: Outline = ()) -> list[Block]:
    """Nest the headings, then walk the document and collect the blocks.

    Fast and deterministic: what the tests exercise, on documents built in code or cached.
    """
    document = document.model_copy(deep=True)  # the heading levels are set in place
    _demote_sentences(document)
    for item in document.texts:
        if isinstance(item, SectionHeaderItem):
            item.level = _UNNUMBERED  # Docling's layout model puts every heading at level 1
    stage = HeadingHierarchyModel(options=HeadingHierarchyOptions(enabled=True, use_style=False))
    stage.assign_heading_levels(document, outline=list(outline))

    blocks: list[Block] = []
    divisions: list[tuple[int, Division]] = []  # (level, division) open, outermost first
    for item, _ in document.iterate_items():
        if isinstance(item, SectionHeaderItem):
            # Read from `orig`: a list item promoted to a heading keeps its number only there.
            _open(divisions, item.level, Division(kind="heading", title=_clean(item.orig)))
            continue
        if isinstance(item, ListItem):
            kind, text = "text", _clean(f"{item.marker} {item.text}")
        elif isinstance(item, TextItem) and item.label != DocItemLabel.FOOTNOTE:
            kind, text = "text", _clean(item.text)
        elif isinstance(item, TableItem) and item.label == DocItemLabel.TABLE:
            kind, text = "table", _markdown_table(item)
        else:  # footnotes, pictures (no OCR, so no text) and the table of contents
            continue
        if kind == "text" and (point := _POINT.match(text)):
            _open(divisions, _POINT_LEVEL, Division(kind="point", number=point["number"]))
        if not text:
            continue
        path = tuple(division for _, division in divisions)
        last = blocks[-1] if blocks else None
        if kind == "text" and last is not None and last.kind == "text" and last.path == path:
            blocks[-1] = last.model_copy(update={"text": f"{last.text}\n{text}"})
        else:
            page = item.prov[0].page_no if item.prov else None
            blocks.append(Block(kind=kind, text=text, path=path, page=page))
    return blocks


def _open(divisions: list[tuple[int, Division]], level: int, division: Division) -> None:
    """Open a division: it closes the open ones at the same level or deeper."""
    while divisions and divisions[-1][0] >= level:
        divisions.pop()
    divisions.append((level, division))


def _demote_sentences(document: DoclingDocument) -> None:
    """Turn into text the headings that end with ":", ";" or "," or start in lower case, and
    the points of a legal text ("3.5.5. The relevant entities shall... procedures.").

    Before the heading stage: "49. Advisable measures:" as a level-1 heading would push the
    chapters after it one level down (EDPB Guidelines 01/2021).
    """
    # Selected first: each replacement renumbers the self_ref of the items after it.
    sentences = [
        item
        for item in document.texts
        if isinstance(item, SectionHeaderItem)
        and (
            (text := _clean(item.orig)).endswith((":", ";", ","))
            or text[:1].islower()
            or (_POINT.match(text) and text.endswith("."))
        )
    ]
    for item in sentences:
        text = TextItem(
            self_ref=item.self_ref,
            parent=item.parent,
            label=DocItemLabel.TEXT,
            text=item.text,
            orig=item.orig,
            prov=item.prov,
        )
        document.replace_item(new_item=text, old_item=item)


def _clean(text: str) -> str:
    """One line, single spaces, Wingdings glyphs replaced by their Unicode equivalent."""
    return " ".join(text.translate(_GLYPHS).split())


def _markdown_table(table: TableItem) -> str:
    """Markdown, first row as header, like the EUR-Lex parser.

    Docling repeats a merged cell in every row and column it covers, so each row still makes
    sense on its own once the chunker splits a long table.
    """
    rows = [[_clean(cell.text).replace("|", r"\|") for cell in row] for row in table.data.grid]
    if not any(any(row) for row in rows):
        return ""
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    lines.insert(1, "| " + " | ".join(["---"] * len(rows[0])) + " |")
    return "\n".join(lines)
