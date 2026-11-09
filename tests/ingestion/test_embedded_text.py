from collections.abc import Sequence

from agentic_rag.document import Chunk, Division
from agentic_rag.ingestion.contextual.prompt import render_document
from agentic_rag.ingestion.embedded_text import ContextualTexts, chunk_texts, titled_texts

_CHAPTER_IV = Division(kind="chapter", number="IV", title="Controller and processor")
_ARTICLE_33 = Division(kind="article", number="33", title="Notification")
_RECITAL_85 = Division(kind="recital", number="85")


def _chunk(text: str, *path: Division) -> Chunk:
    return Chunk(kind="text", text=text, path=path)


class _Fake:
    """Situates a chunk by its text, and keeps the documents it was given."""

    name = "gpt-test"

    def __init__(self) -> None:
        self.documents: list[str] = []

    def situate(self, document: str, chunk: Chunk) -> str:
        self.documents.append(document)
        return f"context of {chunk.text}"


def test_the_baseline_embeds_each_chunk_text_alone() -> None:
    chunks = [_chunk("1. In the case...", _ARTICLE_33), _chunk("(85) A breach...", _RECITAL_85)]

    assert chunk_texts("GDPR", chunks) == ["1. In the case...", "(85) A breach..."]


def test_the_document_title_and_the_titles_come_before_the_text() -> None:
    texts = titled_texts("GDPR", [_chunk("1. In the case...", _CHAPTER_IV, _ARTICLE_33)])

    assert texts == [
        "GDPR > Chapter IV: Controller and processor > Article 33: Notification"
        "\n\n1. In the case..."
    ]


def test_an_untitled_division_still_gives_its_label() -> None:
    assert titled_texts("GDPR", [_chunk("(85) A breach...", _RECITAL_85)]) == [
        "GDPR > Recital 85\n\n(85) A breach..."
    ]


def test_a_chunk_without_a_path_gets_the_document_title_alone() -> None:
    assert titled_texts("ESAs", [_chunk("JC 2024 33")]) == ["ESAs\n\nJC 2024 33"]


def test_without_a_title_or_a_path_the_text_is_embedded_as_is() -> None:
    assert titled_texts("", [_chunk("JC 2024 33")]) == ["JC 2024 33"]


def test_there_is_one_titled_text_per_chunk_in_order() -> None:
    chunks = [_chunk("b", _ARTICLE_33), _chunk("a", _RECITAL_85)]

    assert [text.rsplit("\n\n")[-1] for text in titled_texts("GDPR", chunks)] == ["b", "a"]


def test_the_context_comes_before_the_text() -> None:
    texts = ContextualTexts(_Fake())("GDPR", [_chunk("a", _ARTICLE_33), _chunk("b", _RECITAL_85)])

    assert texts == ["context of a\n\na", "context of b\n\nb"]


def test_the_llm_reads_the_whole_document_with_its_title_for_each_chunk() -> None:
    fake = _Fake()
    chunks: Sequence[Chunk] = [_chunk("a", _ARTICLE_33), _chunk("b", _RECITAL_85)]

    ContextualTexts(fake)("GDPR", chunks)

    assert fake.documents == [render_document("GDPR", chunks)] * 2


def test_a_document_without_chunks_asks_nothing() -> None:
    assert ContextualTexts(_Fake())("GDPR", []) == []
