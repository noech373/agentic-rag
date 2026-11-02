"""Pooling: judge the chunks that several runs rank high and that no judgment covers yet.

The pool of a question is the union of the top `POOL_DEPTH` of each run. A chunk is judged for
one question: the same chunk may be pooled, and judged, for several questions. An LLM pre-judges
the pairs, each verdict appended to `llm.jsonl` as it comes, so that an interrupted pooling resumes.
"""

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentic_rag.evaluation.comparison import IncomparableRunsError
from agentic_rag.evaluation.golden import Question
from agentic_rag.evaluation.judgments import Judgments
from agentic_rag.evaluation.metrics import is_judged
from agentic_rag.evaluation.record import RunRecord
from agentic_rag.evaluation.relevance_judge import RelevanceJudge, Verdict
from agentic_rag.evaluation.runner import JUDGED_K
from agentic_rag.storage.base import StoredChunk

POOL_DEPTH = JUDGED_K  # the top that judged@k checks


class PoolJudgment(BaseModel):
    """One line of llm.jsonl: the LLM's verdict on one chunk for one question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    chunk: str  # "gdpr#12"
    model: str
    verdict: Verdict


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


def not_yet_judged(
    to_judge: Sequence[tuple[Question, StoredChunk]], done: Sequence[PoolJudgment]
) -> list[tuple[Question, StoredChunk]]:
    """The pairs the LLM has no verdict for yet: what an interrupted pooling still has to do."""
    judged = {(j.question_id, j.chunk) for j in done}
    return [(q, c) for q, c in to_judge if (q.id, c.key) not in judged]


def judge_pool(
    to_judge: Sequence[tuple[Question, StoredChunk]],
    judge: RelevanceJudge,
    path: Path,
    *,
    workers: int = 8,
) -> list[PoolJudgment]:
    """Judge the pairs in parallel, appending each verdict to `path` as soon as it comes.

    If a call fails, the verdicts already received stay in the file and the rest is cancelled:
    running again judges only what is missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    judged: list[PoolJudgment] = []
    with ThreadPoolExecutor(workers) as executor, path.open("a", encoding="utf-8") as file:
        futures = {executor.submit(judge.judge, q, c): (q, c) for q, c in to_judge}
        try:
            for future in as_completed(futures):  # only this thread writes to the file
                question, chunk = futures[future]
                judgment = PoolJudgment(
                    question_id=question.id,
                    chunk=chunk.key,
                    model=judge.model,
                    verdict=future.result(),
                )
                file.write(judgment.model_dump_json() + "\n")
                file.flush()
                judged.append(judgment)
        except BaseException:  # also Ctrl+C
            executor.shutdown(cancel_futures=True)
            raise
    return judged


def load_pool_judgments(path: Path) -> list[PoolJudgment]:
    """The verdicts saved so far; none before the first pooling."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PoolJudgment.model_validate_json(line) for line in lines if line.strip()]
