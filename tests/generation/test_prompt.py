from agentic_rag.document import Chunk, Division, IndexedDocument, RetrievedChunk
from agentic_rag.generation.base import NO_ANSWER
from agentic_rag.generation.prompt import INSTRUCTIONS, build_input

_CHAPTER_IV = Division(kind="chapter", number="IV", title="Controller and processor")
_SECTION_2 = Division(kind="section", number="2", title="Security of personal data")
_ARTICLE_33 = Division(kind="article", number="33", title="Notification of a personal data breach")


def _source(
    *path: Division,
    text: str = "1. In the case of a personal data breach, ...",
    page: int | None = None,
    short_title: str = "GDPR",
) -> RetrievedChunk:
    document = IndexedDocument(
        id="gdpr",
        title="Regulation (EU) 2016/679 (General Data Protection Regulation)",
        short_title=short_title,
        source_url="https://data.europa.eu/eli/reg/2016/679/oj",
        regulations=("GDPR",),
        groups=("public",),
        sha256="0" * 64,
        embedding_model="fake@2",
    )
    chunk = Chunk(kind="text", text=text, path=path, page=page)
    return RetrievedChunk(document=document, position=0, chunk=chunk, score=1.0)


def test_the_instructions_give_the_exact_no_answer_sentence() -> None:
    assert f'reply exactly: "{NO_ANSWER}"' in INSTRUCTIONS


def test_the_sources_are_numbered_from_one_and_the_question_comes_last() -> None:
    sources = [_source(text="First."), _source(text="Second.")]

    prompt = build_input("How long?", sources)

    assert prompt.index('<source id="1">') < prompt.index("First.")
    assert prompt.index('<source id="2">') < prompt.index("Second.")
    assert prompt.endswith("<question>\nHow long?\n</question>")


def test_a_source_says_where_it_comes_from_with_the_titles() -> None:
    prompt = build_input("How long?", [_source(_CHAPTER_IV, _SECTION_2, _ARTICLE_33)])

    assert prompt.startswith(
        '<source id="1">\n'
        "<location>Regulation (EU) 2016/679 (General Data Protection Regulation), "
        "Chapter IV: Controller and processor > Section 2: Security of personal data > "
        "Article 33: Notification of a personal data breach</location>\n"
        "1. In the case of a personal data breach, ...\n"
        "</source>"
    )


def test_the_location_of_a_pdf_source_ends_with_its_page() -> None:
    heading = Division(kind="heading", title="1. Article 33 requirements")

    prompt = build_input("How long?", [_source(heading, page=10)])

    assert "1. Article 33 requirements, p. 10</location>" in prompt
