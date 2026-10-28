from agentic_rag.document import Chunk
from agentic_rag.evaluation.golden import Evidence, GoldenSet, Question, Quote
from agentic_rag.evaluation.runner import MRR_K, RECALL_KS, run_retriever, tuning_questions
from agentic_rag.storage.base import IndexedDocument, RetrievedChunk


def _question(
    id: str = "q",
    quote: str | None = "notify without undue delay",
    holdout: bool = False,
) -> Question:
    evidence = (Evidence(any_of=(Quote(document="gdpr", quote=quote),)),) if quote else ()
    return Question(
        id=id,
        category="factual" if quote else "unanswerable",
        origin="blind",
        holdout=holdout,
        question="When to notify?",
        answer="!",
        evidence=evidence,
    )


def _retrieved(position: int, text: str) -> RetrievedChunk:
    document = IndexedDocument(
        id="gdpr",
        title="GDPR",
        short_title="GDPR",
        source_url=None,
        regulations=(),
        groups=("public",),
        sha256="0",
        embedding_model="m",
    )
    return RetrievedChunk(
        document=document,
        position=position,
        chunk=Chunk(kind="text", text=text, path=()),
        score=1.0,
    )


class _FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self.results


def test_tuning_questions_skip_held_out_and_unanswerable() -> None:
    golden_set = GoldenSet(
        questions=(
            _question("kept"),
            _question("held-out", holdout=True),
            _question("unanswerable", quote=None),
        )
    )

    assert [q.id for q in tuning_questions(golden_set)] == ["kept"]


def test_the_retriever_is_called_once_per_question_with_the_deepest_k() -> None:
    retriever = _FakeRetriever([])

    run_retriever([_question("a"), _question("b")], retriever)

    assert retriever.calls == [("When to notify?", max(RECALL_KS))] * 2


def test_each_k_is_scored_on_the_same_ranking() -> None:
    filler = [_retrieved(i, f"unrelated {i}") for i in range(6)]
    hit = _retrieved(6, "Notify without undue delay.")
    retriever = _FakeRetriever([*filler, hit])

    [result] = run_retriever([_question()], retriever)

    assert result.recall == {"recall@5": 0, "recall@8": 1, "recall@20": 1, "recall@50": 1}
    assert result.mrr == 1 / 7
    assert result.question_id == "q"
    assert result.category == "factual"


def test_the_mrr_ignores_chunks_beyond_its_k() -> None:
    filler = [_retrieved(i, f"unrelated {i}") for i in range(MRR_K)]
    retriever = _FakeRetriever([*filler, _retrieved(MRR_K, "Notify without undue delay.")])

    [result] = run_retriever([_question()], retriever)

    assert result.mrr == 0


def test_the_retrieved_chunks_are_kept_by_document_and_position() -> None:
    retriever = _FakeRetriever([_retrieved(3, "x"), _retrieved(7, "y")])

    [result] = run_retriever([_question()], retriever)

    assert result.retrieved == ("gdpr#3", "gdpr#7")
