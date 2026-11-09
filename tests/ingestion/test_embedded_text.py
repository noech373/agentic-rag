from agentic_rag.document import Chunk, Division
from agentic_rag.ingestion.embedded_text import chunk_texts, titled_texts

_CHAPTER_IV = Division(kind="chapter", number="IV", title="Controller and processor")
_ARTICLE_33 = Division(kind="article", number="33", title="Notification")
_RECITAL_85 = Division(kind="recital", number="85")


def _chunk(text: str, *path: Division) -> Chunk:
    return Chunk(kind="text", text=text, path=path)


def test_the_baseline_embeds_each_chunk_text_alone() -> None:
    chunks = [_chunk("1. In the case...", _ARTICLE_33), _chunk("(85) A breach...", _RECITAL_85)]

    assert chunk_texts("GDPR", chunks) == ["1. In the case...", "(85) A breach..."]


def test_the_titles_come_before_the_text() -> None:
    texts = titled_texts("GDPR", [_chunk("1. In the case...", _CHAPTER_IV, _ARTICLE_33)])

    assert texts == [
        "Chapter IV: Controller and processor > Article 33: Notification\n\n1. In the case..."
    ]


def test_an_untitled_division_still_gives_its_label() -> None:
    assert titled_texts("GDPR", [_chunk("(85) A breach...", _RECITAL_85)]) == [
        "Recital 85\n\n(85) A breach..."
    ]


def test_a_chunk_without_a_path_is_embedded_as_is() -> None:
    assert titled_texts("ESAs", [_chunk("JC 2024 33")]) == ["JC 2024 33"]


def test_there_is_one_text_per_chunk_in_order() -> None:
    chunks = [_chunk("b", _ARTICLE_33), _chunk("a", _RECITAL_85)]

    assert [text.rsplit("\n\n")[-1] for text in titled_texts("GDPR", chunks)] == ["b", "a"]
