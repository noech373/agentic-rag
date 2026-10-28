"""Run a retriever on the golden set: one result per question, ready to aggregate and to save."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from agentic_rag.evaluation.golden import Category, GoldenSet, Question
from agentic_rag.evaluation.metrics import mrr_at_k, recall_at_k
from agentic_rag.retrieval.base import Retriever

RECALL_KS = (5, 8, 20, 50)
MRR_K = 10


class QuestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str
    category: Category
    recall: dict[str, float]
    mrr: float
    retrieved: tuple[str, ...]  # "document_id#position", best first, to inspect and pool


def tuning_questions(golden_set: GoldenSet) -> tuple[Question, ...]:
    """The questions to tune on: not held out, and with facts to find (so not unanswerable)."""
    return tuple(q for q in golden_set.questions if not q.holdout and q.evidence)


def run_retriever(questions: Sequence[Question], retriever: Retriever) -> list[QuestionResult]:
    """Retrieve once per question, deep enough for the largest k, and score every k on it."""
    depth = max((*RECALL_KS, MRR_K))
    results: list[QuestionResult] = []
    for question in questions:
        retrieved = retriever.retrieve(question.question, depth)
        chunks = [r.chunk for r in retrieved]
        results.append(
            QuestionResult(
                question_id=question.id,
                category=question.category,
                recall={f"recall@{k}": recall_at_k(question, chunks, k) for k in RECALL_KS},
                mrr=mrr_at_k(question, chunks, MRR_K),
                retrieved=tuple(f"{r.document.id}#{r.position}" for r in retrieved),
            )
        )
    return results
