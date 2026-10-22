import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

from agentic_rag.document import Block, Division
from agentic_rag.ingestion.markdown import parse_markdown

_CORPUS = Path(__file__).parents[1] / "data" / "corpus"


def _parse(text: str) -> list[Block]:
    return parse_markdown(text.encode())


def _paths(blocks: list[Block]) -> list[str]:
    return [" > ".join(division.title or "" for division in block.path) for block in blocks]


# --- Headings ---


def test_headings_give_the_path() -> None:
    blocks = _parse(
        "# Sheet n°2: Prepare your development\n"
        "## Methodological choices\n"
        "a\n"
        "## Technological choices\n"
        "#### Architecture and features\n"
        "b\n"
        "#### Tools and practices ##\n"
        "c\n"
    )

    assert _paths(blocks) == [
        "Sheet n°2: Prepare your development > Methodological choices",
        "Sheet n°2: Prepare your development > Technological choices > Architecture and features",
        "Sheet n°2: Prepare your development > Technological choices > Tools and practices",
    ]
    assert blocks[0].path[0] == Division(
        kind="heading", title="Sheet n°2: Prepare your development"
    )


def test_the_lead_paragraph_styled_as_a_heading_is_text() -> None:
    blocks = _parse(
        "# Sheet n°13: Prepare for the exercise of people rights\n"
        "\n"
        "#### The persons whose data you process have rights on their data.\n"
        "\n"
        "#### Preparing in advance will enable you to manage them effectively.\n"
        "\n"
        "## Minimum measures to be put in place\n"
        "* a\n"
    )

    assert [(path, block.text) for path, block in zip(_paths(blocks), blocks, strict=True)] == [
        (
            "Sheet n°13: Prepare for the exercise of people rights",
            "The persons whose data you process have rights on their data.\n"
            "Preparing in advance will enable you to manage them effectively.",
        ),
        (
            "Sheet n°13: Prepare for the exercise of people rights"
            " > Minimum measures to be put in place",
            "* a",
        ),
    ]


def test_text_before_any_heading_has_an_empty_path() -> None:
    assert _parse("Introduction.\n# Title\n") == [Block(kind="text", text="Introduction.", path=())]


def test_a_hash_without_space_is_not_a_heading() -> None:
    assert _parse("# Title\n#hashtag\n")[0].text == "#hashtag"


# --- Text ---


def test_the_text_under_a_heading_is_one_block_one_line_per_paragraph_or_item() -> None:
    blocks = _parse(
        "# Title\n"
        "## Hosting\n"
        "\n"
        "* **Choose a service provider**  that is transparent.\n"
        "\n"
        "* Other points to be aware of include:\n"
        "    - the existence of an accessible security policy;\n"
        "        * nested deeper\n"
        "\n"
        "\n"
        "1. first step\n"
    )

    assert blocks == [
        Block(
            kind="text",
            text=(
                "* **Choose a service provider** that is transparent.\n"
                "* Other points to be aware of include:\n"
                "    - the existence of an accessible security policy;\n"
                "        * nested deeper\n"
                "1. first step"
            ),
            path=(
                Division(kind="heading", title="Title"),
                Division(kind="heading", title="Hosting"),
            ),
        )
    ]


def test_links_are_reduced_to_their_text() -> None:
    blocks = _parse(
        "# Sheet n°15: [Article 6](https://eur-lex.europa.eu/eli/reg/2016/679/oj)\n"
        "The CNIL provides a [map of data protection](https://www.cnil.fr/en/map) and [](x).\n"
    )

    assert blocks[0].path[0].title == "Sheet n°15: Article 6"
    assert blocks[0].text == "The CNIL provides a map of data protection and ."


# --- Tables ---


def test_a_table_is_a_block_of_its_own_with_normalised_cells() -> None:
    blocks = _parse(
        "# Legal basis\n"
        "* The following table summarizes the rights:\n"
        "\n"
        "| | Right of access | Right to object|\n"
        "|:-----:|:---:|---|\n"
        "| **Consent**      | ✔             | **Withdraw of consent**|\n"
        "\n"
        "* The legal basis must always appear.\n"
    )

    assert [(block.kind, block.text) for block in blocks] == [
        ("text", "* The following table summarizes the rights:"),
        (
            "table",
            "|  | Right of access | Right to object |\n"
            "| --- | --- | --- |\n"
            "| **Consent** | ✔ | **Withdraw of consent** |",
        ),
        ("text", "* The legal basis must always appear."),
    ]
    assert {block.path for block in blocks} == {(Division(kind="heading", title="Legal basis"),)}


def test_a_blank_line_ends_a_table() -> None:
    blocks = _parse("# T\n| a |\n|---|\n| 1 |\n\n| b |\n|---|\n| 2 |\n")

    assert [block.text.splitlines()[0] for block in blocks] == ["| a |", "| b |"]


# --- On the real corpus (skipped until `uv run agentic-rag download`) ---

_SHEETS = [f"cnil-developer-guide-{number:02}" for number in range(17)]


def _sheet(document_id: str) -> str:
    path = _CORPUS / f"{document_id}.md"
    if not path.exists():
        pytest.skip(f"{path} not found: run `uv run agentic-rag download`")
    return path.read_text()


@pytest.mark.parametrize("document_id", _SHEETS)
def test_corpus_every_block_is_under_the_sheet_title(document_id: str) -> None:
    number = int(document_id[-2:])
    blocks = _parse(_sheet(document_id))

    assert blocks
    for block in blocks:
        assert re.match(rf"Sheet n°0?{number}: ", block.path[0].title or "")  # "Sheet n°09: ..."
        assert block.text.strip()
        assert "](" not in block.text


@pytest.mark.parametrize("document_id", _SHEETS)
def test_corpus_only_tables_split_a_division(document_id: str) -> None:
    for previous, block in pairwise(_parse(_sheet(document_id))):
        if previous.path == block.path:
            assert "table" in (previous.kind, block.kind)


def test_corpus_the_lead_paragraphs_are_text() -> None:
    blocks = _parse(_sheet("cnil-developer-guide-13"))

    assert len(blocks[0].path) == 1
    assert blocks[0].text.startswith("The persons whose data you process have rights")
    assert blocks[0].text.splitlines()[1].startswith("Preparing in advance how they will contact")


def test_corpus_the_rights_table() -> None:
    (table,) = [
        block for block in _parse(_sheet("cnil-developer-guide-15")) if block.kind == "table"
    ]
    rows = table.text.splitlines()

    assert len(rows) == 8
    assert rows[1] == "| --- | --- | --- | --- | --- | --- | --- |"
    assert rows[3] == "| **Contract** | ✔ | ✔ | ✔ | ✔ | ✔ | ✘ |"


def _words(text: str | None) -> Counter[str]:
    return Counter(re.findall(r"\w+", (text or "").lower()))


@pytest.mark.parametrize("document_id", _SHEETS)
def test_corpus_no_text_is_lost_nor_invented(document_id: str) -> None:
    """The words of the sheet, link URLs aside, are those of the blocks and of the headings.

    Each heading counts once: as the last division of a path, for every path in the sheet.
    """
    source = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", _sheet(document_id))
    blocks = _parse(_sheet(document_id))
    found: Counter[str] = Counter()
    for block in blocks:
        found += _words(block.text)
    for path in {block.path[:depth] for block in blocks for depth in range(1, len(block.path) + 1)}:
        found += _words(path[-1].title)

    assert found == _words(source)
