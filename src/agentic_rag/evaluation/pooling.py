"""Pooling: judge the chunks that several runs rank high and that no judgment covers yet.

The pool of a question is the union of the top `POOL_DEPTH` of each run. A chunk is judged for
one question: the same chunk may be pooled, and judged, for several questions.
"""

from collections.abc import Mapping, Sequence

from agentic_rag.evaluation.comparison import IncomparableRunsError
from agentic_rag.evaluation.golden import Question
from agentic_rag.evaluation.judgments import Judgments
from agentic_rag.evaluation.metrics import is_judged
from agentic_rag.evaluation.record import RunRecord
from agentic_rag.evaluation.runner import JUDGED_K
from agentic_rag.storage.base import StoredChunk

POOL_DEPTH = JUDGED_K  # the top that judged@k checks


def pool(question_id: str, runs: Sequence[RunRecord], depth: int = POOL_DEPTH) -> list[str]:
    """The first `depth` chunks of each run for the question, each once, in the order first seen."""
    pooled: dict[str, None] = {}  # a dict keeps the insertion order
    for run in runs:
        for result in run.results:
            if result.question_id == question_id:
                pooled.update(dict.fromkeys(result.retrieved[:depth]))
    return list(pooled)


def chunks_to_judge(
    questions: Sequence[Question],
    runs: Sequence[RunRecord],
    judgments: Judgments,
    chunks: Mapping[str, StoredChunk],
) -> list[tuple[Question, StoredChunk]]:
    """The pooled (question, chunk) pairs that neither the golden set nor the judgments cover.

    Questions absent from the runs (the holdout) have an empty pool.
    """
    if len({run.golden_sha256 for run in runs}) > 1:
        raise IncomparableRunsError("the runs used different golden sets: rerun them")

    to_judge: list[tuple[Question, StoredChunk]] = []
    for question in questions:
        rejected = judgments.get(question.id, ())
        for key in pool(question.id, runs):
            if key not in chunks:
                raise ValueError(f"{key} is not in the index: it changed since the runs")
            chunk = chunks[key]
            if not is_judged(question, rejected, chunk.chunk):
                to_judge.append((question, chunk))
    return to_judge
