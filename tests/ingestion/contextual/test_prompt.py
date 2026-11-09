from agentic_rag.document import Chunk, Division
from agentic_rag.ingestion.contextual.prompt import INSTRUCTIONS, build_input, render_document

_CHAPTER_IV = Division(kind="chapter", number="IV", title="Controller and processor")
_ARTICLE_33 = Division(kind="article", number="33", title="Notification of a personal data breach")
_ARTICLE_34 = Division(kind="article", number="34", title="Communication to the data subject")
_RECITAL_85 = Division(kind="recital", number="85")


def _chunk(text: str, *path: Division) -> Chunk:
    return Chunk(kind="text", text=text, path=path)


def test_a_chunk_comes_after_its_titled_path() -> None:
    rendered = render_document("GDPR", [_chunk("1. In the case...", _CHAPTER_IV, _ARTICLE_33)])

    assert rendered == (
        "<document>\n"
        "<title>GDPR</title>\n\n"
        "## Chapter IV: Controller and processor > "
        "Article 33: Notification of a personal data breach\n\n"
        "1. In the case...\n"
        "</document>"
    )


def test_chunks_stay_in_the_given_order() -> None:
    rendered = render_document(
        "GDPR", [_chunk("second", _ARTICLE_34), _chunk("first", _ARTICLE_33)]
    )

    assert rendered.index("second") < rendered.index("first")


def test_consecutive_chunks_with_the_same_path_share_one_heading() -> None:
    rendered = render_document(
        "GDPR", [_chunk("1. ...", _ARTICLE_33), _chunk("2. ...", _ARTICLE_33)]
    )

    assert rendered.count("## ") == 1
    assert "1. ...\n\n2. ..." in rendered


def test_a_new_path_gets_its_own_heading() -> None:
    rendered = render_document("GDPR", [_chunk("a", _ARTICLE_33), _chunk("b", _ARTICLE_34)])

    assert "## Article 33: Notification of a personal data breach\n\na" in rendered
    assert "## Article 34: Communication to the data subject\n\nb" in rendered


def test_a_path_seen_again_after_another_is_repeated() -> None:
    rendered = render_document(
        "GDPR",
        [_chunk("a", _ARTICLE_33), _chunk("b", _ARTICLE_34), _chunk("c", _ARTICLE_33)],
    )

    assert rendered.count("## Article 33") == 2


def test_a_chunk_without_a_path_has_no_heading() -> None:
    rendered = render_document("ESAs", [_chunk("JC 2024 33"), _chunk("(1) ...", _RECITAL_85)])

    assert rendered == (
        "<document>\n<title>ESAs</title>\n\nJC 2024 33\n\n## Recital 85\n\n(1) ...\n</document>"
    )


def test_a_path_seen_again_after_a_chunk_without_a_path_is_repeated() -> None:
    rendered = render_document(
        "ESAs", [_chunk("a", _RECITAL_85), _chunk("b"), _chunk("c", _RECITAL_85)]
    )

    assert rendered.count("## Recital 85") == 2


def test_the_instructions_ask_for_the_context_alone() -> None:
    assert "only with a succinct context" in INSTRUCTIONS
    assert "without repeating the passage" in INSTRUCTIONS


def test_the_instructions_keep_the_context_on_its_passage_and_name_the_document() -> None:
    assert "Do not summarize the rest of the document" in INSTRUCTIONS
    assert "Start by naming the document" in INSTRUCTIONS


def test_the_input_is_the_document_then_the_located_passage() -> None:
    built = build_input("<document>...</document>", _chunk("1. In the case...", _ARTICLE_33))

    assert built == (
        "<document>...</document>\n\n"
        "<passage>\n"
        "<location>Article 33: Notification of a personal data breach</location>\n"
        "1. In the case...\n"
        "</passage>"
    )


def test_a_passage_without_a_path_has_no_location() -> None:
    built = build_input("<document>...</document>", _chunk("JC 2024 33"))

    assert built == "<document>...</document>\n\n<passage>\nJC 2024 33\n</passage>"
