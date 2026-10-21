"""Parser for Markdown documents: the CNIL GDPR Developer Guide and the internal policies.

Headings (`#` to `######`) give the path. As for the other formats, a block is the content of
one division, one line per paragraph or list item (nested items keep their indentation), and
a table is a block of its own. Reading line by line is enough: the sheets have no code block,
HTML, image or front matter, and no paragraph wrapped over several lines.

One CNIL habit: a sheet's lead paragraph is styled as a heading right under the title
("#### When designing the architecture of your application, you must..."). As a heading it
would hold no content, since the next "##" closes it, and its text would be lost: it is text.
"""

import re

from agentic_rag.ingestion.document import Block, Division

_HEADING = re.compile(r"(?P<marks>#{1,6})\s+(?P<title>.*?)[\s#]*")  # "## Title", "## Title ##"
_LINK = re.compile(r"\[(?P<text>[^\]]*)\]\([^)]*\)")  # "[text](https://...)": the URL is noise
_SEPARATOR = re.compile(r":?-+:?")  # a cell of the row under a table header: "---", ":---:"


def parse_markdown(content: bytes) -> list[Block]:
    """Split a Markdown document into blocks, each one tagged with its headings."""
    blocks: list[Block] = []
    headings: list[tuple[int, Division]] = []  # (level, heading) open, outermost first
    previous = ""  # the line before: a table row right after another one continues the table
    for line in content.decode().splitlines():
        if heading := _HEADING.fullmatch(line):
            level = len(heading["marks"])
            if level <= 2 or len(headings) != 1:  # not a lead paragraph under the title
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, Division(kind="heading", title=_clean(heading["title"]))))
                previous = line
                continue
            line = heading["title"]
        if line.strip():
            kind = "table" if line.lstrip().startswith("|") else "text"
            text = _table_row(line) if kind == "table" else _indent(line) + _clean(line)
            path = tuple(division for _, division in headings)
            last = blocks[-1] if blocks else None
            continues = kind == "text" or previous.lstrip().startswith("|")
            if last is not None and last.kind == kind and last.path == path and continues:
                blocks[-1] = last.model_copy(update={"text": f"{last.text}\n{text}"})
            else:
                blocks.append(Block(kind=kind, text=text, path=path))
        previous = line
    return blocks


def _table_row(line: str) -> str:
    """A table row with single spaces around its cells; the header's underline as "---"."""
    cells = [_clean(cell) for cell in line.strip().strip("|").split("|")]
    if all(_SEPARATOR.fullmatch(cell) for cell in cells):
        cells = ["---"] * len(cells)
    return "| " + " | ".join(cells) + " |"


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _clean(text: str) -> str:
    """One line, single spaces, links reduced to their text."""
    return " ".join(_LINK.sub(r"\g<text>", text).split())
