from agentic_rag.citation import citation, titled_path
from agentic_rag.document import Chunk, Division, IndexedDocument, StoredChunk

_CHAPTER_IV = Division(kind="chapter", number="IV", title="Controller and processor")
_SECTION_2 = Division(kind="section", number="2", title="Security of personal data")
_ARTICLE_33 = Division(kind="article", number="33", title="Notification of a personal data breach")


def _source(*path: Division, page: int | None = None, short_title: str = "GDPR") -> StoredChunk:
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
    chunk = Chunk(kind="text", text="Text.", path=path, page=page)
    return StoredChunk(document=document, position=0, chunk=chunk)


def test_an_article_is_cited_without_its_chapter_and_section() -> None:
    assert citation(_source(_CHAPTER_IV, _SECTION_2, _ARTICLE_33)) == "GDPR, Article 33"


def test_a_paragraph_is_cited_after_its_article() -> None:
    paragraph = Division(kind="paragraph", number="1")

    assert citation(_source(_CHAPTER_IV, _ARTICLE_33, paragraph)) == "GDPR, Article 33(1)"


def test_recitals_and_annexes_are_cited_by_number() -> None:
    recital = Division(kind="recital", number="85")
    annex = Division(kind="annex", number="III", title="High-risk AI systems")

    assert citation(_source(recital)) == "GDPR, Recital 85"
    assert citation(_source(annex, short_title="AI Act")) == "AI Act, Annex III"


def test_a_guideline_is_cited_by_its_last_heading_and_page() -> None:
    path = (
        Division(kind="heading", title="II. Article 33 - Notification"),
        Division(kind="heading", title="1. Article 33 requirements"),
    )

    assert (
        citation(_source(*path, page=10, short_title="EDPB Guidelines 9/2022"))
        == "EDPB Guidelines 9/2022, 1. Article 33 requirements, p. 10"
    )


def test_headings_below_a_point_do_not_replace_it() -> None:
    path = (
        Division(kind="heading", title="3 INCIDENT HANDLING"),
        Division(kind="point", number="3.2.4"),
        Division(kind="heading", title="GUIDANCE"),
    )

    assert citation(_source(*path, page=30, short_title="ENISA")) == "ENISA, Point 3.2.4, p. 30"


def test_a_chunk_without_path_is_cited_by_its_document() -> None:
    assert citation(_source()) == "GDPR"
    assert citation(_source(page=1)) == "GDPR, p. 1"


def test_a_titled_path_joins_its_divisions_outermost_first() -> None:
    chunk = _source(_CHAPTER_IV, _SECTION_2, _ARTICLE_33).chunk

    assert titled_path(chunk) == (
        f"Chapter IV: {_CHAPTER_IV.title} > Section 2: {_SECTION_2.title} > "
        f"Article 33: {_ARTICLE_33.title}"
    )


def test_a_titled_path_is_empty_without_divisions() -> None:
    assert titled_path(_source().chunk) == ""
