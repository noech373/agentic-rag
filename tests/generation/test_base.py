from agentic_rag.document import Chunk
from agentic_rag.generation.base import NO_ANSWER, Answer
from agentic_rag.storage.base import IndexedDocument, RetrievedChunk


def _sources(count: int) -> tuple[RetrievedChunk, ...]:
    document = IndexedDocument(
        id="gdpr",
        title="GDPR",
        short_title="GDPR",
        source_url=None,
        regulations=("GDPR",),
        groups=("public",),
        sha256="0" * 64,
        embedding_model="fake@2",
    )
    return tuple(
        RetrievedChunk(
            document=document,
            position=i,
            chunk=Chunk(kind="text", text=f"Chunk {i}.", path=()),
            score=1.0,
        )
        for i in range(count)
    )


def _cited(text: str, count: int = 4) -> list[int]:
    return list(Answer(text=text, sources=_sources(count)).cited)


def test_cited_maps_each_number_to_its_source() -> None:
    sources = _sources(3)

    answer = Answer(text="Within 72 hours [1], unless delayed [3].", sources=sources)

    assert answer.cited == {1: sources[0], 3: sources[2]}


def test_cited_numbers_come_once_and_in_order() -> None:
    assert _cited("A [3][1]. B [1].") == [1, 3]


def test_a_list_of_numbers_in_one_bracket_is_read_too() -> None:
    assert _cited("A [1, 3]. B [2,4].") == [1, 2, 3, 4]


def test_numbers_without_a_source_are_ignored() -> None:
    assert _cited("A [0]. B [5]. C [2].", count=4) == [2]


def test_an_answer_without_brackets_cites_nothing() -> None:
    assert _cited(NO_ANSWER) == []
    assert _cited("Article 33 GDPR, 72 hours") == []
