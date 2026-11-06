from pathlib import Path

import pytest

from agentic_rag.document import Chunk, Division
from agentic_rag.ingestion.contextual import cached
from agentic_rag.ingestion.contextual.cached import CachedContext, CachedContextualizer

_DOCUMENT = "<document>...</document>"
_ARTICLE_33 = Division(kind="article", number="33", title="Notification")
_ARTICLE_34 = Division(kind="article", number="34", title="Communication")
_CHUNK = Chunk(kind="text", text="Not later than 72 hours.", path=(_ARTICLE_33,))


class _Fake:
    """Answers with a new text at each call, as an LLM may, and can fail after some calls."""

    def __init__(self, name: str = "gpt-test", fail_after: int | None = None) -> None:
        self.name = name
        self.calls = 0
        self._fail_after = fail_after

    def situate(self, document: str, chunk: Chunk) -> str:
        if self.calls == self._fail_after:
            raise RuntimeError("the LLM failed")
        self.calls += 1
        return f"context {self.calls}"


def _lines(path: Path) -> list[CachedContext]:
    return [
        CachedContext.model_validate_json(line) for line in path.read_text("utf-8").splitlines()
    ]


def test_a_new_request_asks_the_contextualizer(tmp_path: Path) -> None:
    fake = _Fake()

    context = CachedContextualizer(fake, tmp_path / "c.jsonl").situate(_DOCUMENT, _CHUNK)

    assert context == "context 1"
    assert fake.calls == 1


def test_the_same_request_is_not_asked_twice(tmp_path: Path) -> None:
    fake = _Fake()
    contextualizer = CachedContextualizer(fake, tmp_path / "c.jsonl")

    contextualizer.situate(_DOCUMENT, _CHUNK)
    context = contextualizer.situate(_DOCUMENT, _CHUNK)

    assert context == "context 1"
    assert fake.calls == 1


def test_a_new_ingestion_reads_the_contexts_of_the_previous_one(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    CachedContextualizer(_Fake(), path).situate(_DOCUMENT, _CHUNK)
    fake = _Fake()

    context = CachedContextualizer(fake, path).situate(_DOCUMENT, _CHUNK)

    assert context == "context 1"
    assert fake.calls == 0


def test_a_failure_keeps_the_contexts_already_received_and_stores_nothing_for_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "c.jsonl"
    contextualizer = CachedContextualizer(_Fake(fail_after=1), path)
    contextualizer.situate(_DOCUMENT, _CHUNK)

    with pytest.raises(RuntimeError):
        contextualizer.situate("<document>other</document>", _CHUNK)

    assert len(_lines(path)) == 1


def test_a_cached_context_adds_no_line(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    contextualizer = CachedContextualizer(_Fake(), path)

    contextualizer.situate(_DOCUMENT, _CHUNK)
    contextualizer.situate(_DOCUMENT, _CHUNK)

    assert len(_lines(path)) == 1


def test_another_model_asks_again(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    CachedContextualizer(_Fake("gpt-a"), path).situate(_DOCUMENT, _CHUNK)
    fake = _Fake("gpt-b")

    CachedContextualizer(fake, path).situate(_DOCUMENT, _CHUNK)

    assert fake.calls == 1


def test_other_instructions_ask_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "c.jsonl"
    CachedContextualizer(_Fake(), path).situate(_DOCUMENT, _CHUNK)
    monkeypatch.setattr(cached, "INSTRUCTIONS", "Stricter instructions.")
    fake = _Fake()

    CachedContextualizer(fake, path).situate(_DOCUMENT, _CHUNK)

    assert fake.calls == 1


@pytest.mark.parametrize(
    ("document", "chunk"),
    [
        ("<document>other</document>", _CHUNK),
        (_DOCUMENT, Chunk(kind="text", text="Another text.", path=(_ARTICLE_33,))),
        (_DOCUMENT, Chunk(kind="text", text=_CHUNK.text, path=(_ARTICLE_34,))),
    ],
    ids=["document", "text", "location"],
)
def test_another_request_asks_again(tmp_path: Path, document: str, chunk: Chunk) -> None:
    fake = _Fake()
    contextualizer = CachedContextualizer(fake, tmp_path / "c.jsonl")
    contextualizer.situate(_DOCUMENT, _CHUNK)

    contextualizer.situate(document, chunk)

    assert fake.calls == 2


def test_a_line_can_be_read_without_the_document(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    long_text = "é" * 100

    CachedContextualizer(_Fake(), path).situate(
        _DOCUMENT, Chunk(kind="text", text=long_text, path=(_ARTICLE_33,))
    )

    line = _lines(path)[0]
    assert line.model == "gpt-test"
    assert line.location == "Article 33: Notification"
    assert line.passage == "é" * 80
    assert line.context == "context 1"


def test_it_keeps_the_name_of_its_contextualizer(tmp_path: Path) -> None:
    assert CachedContextualizer(_Fake("gpt-a"), tmp_path / "c.jsonl").name == "gpt-a"
