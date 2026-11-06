"""How to refer to a chunk: in the answers, the traces, the judge's prompt and the reviews."""

from agentic_rag.document import Chunk, Division, DivisionKind, StoredChunk

_NAMES: dict[DivisionKind, str] = {
    "chapter": "Chapter",
    "section": "Section",
    "article": "Article",
    "paragraph": "Paragraph",
    "recital": "Recital",
    "annex": "Annex",
    "point": "Point",
    "heading": "",
}
_UNITS: frozenset[DivisionKind] = frozenset({"article", "recital", "annex", "point"})


def citation(source: StoredChunk) -> str:
    """How the answer refers to a source: "GDPR, Article 33(1)", "EDPB Guidelines 9/2022, p. 10".

    The most precise legal unit (article and paragraph, recital, annex, point), or else the last
    heading: chapters and sections add nothing once the article is known.
    """
    reference: str | None = None
    in_unit = False
    for division in source.chunk.path:
        if division.kind in _UNITS:
            reference, in_unit = label(division), True
        elif division.kind == "paragraph" and reference is not None:
            reference += f"({division.number})"
        elif division.kind == "heading" and not in_unit:
            reference = division.title
    return with_page(source, [source.document.short_title, reference])


def label(division: Division, *, titled: bool = False) -> str:
    """The name of a division: "Article 33", or with its title "Article 33: Notification of..."."""
    name = " ".join(part for part in (_NAMES[division.kind], division.number) if part)
    if titled and division.title:
        return f"{name}: {division.title}" if name else division.title
    return name or (division.title or "")


def titled_path(chunk: Chunk) -> str:
    """Where a chunk is, with titles: "Chapter IV: Controller... > Article 33: Notification..."."""
    return " > ".join(label(division, titled=True) for division in chunk.path)


def with_page(source: StoredChunk, parts: list[str | None]) -> str:
    """The parts that are set, then the page of the chunk if it has one."""
    if source.chunk.page is not None:
        parts.append(f"p. {source.chunk.page}")
    return ", ".join(part for part in parts if part)
