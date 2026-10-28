from collections.abc import Sequence

from agentic_rag.document import Chunk
from agentic_rag.evaluation.golden import Question, contains, normalize


def recall_at_k(question: Question, chunks: Sequence[Chunk], k: int) -> float:
    """The share of the question's facts that the first `k` chunks state."""
    if not question.evidence:
        raise ValueError(f"{question.id} has no facts: its recall is undefined")

    top_texts = [normalize(chunk.text) for chunk in chunks[:k]]

    found = 0
    for evidence in question.evidence:
        for quote in evidence.any_of:
            if any(contains(text, quote.quote) for text in top_texts):
                found += 1
                break

    return found / len(question.evidence)


def mrr_at_k(question: Question, chunks: Sequence[Chunk], k: int) -> float:
    """1 / the rank of the first of the first `k` chunks that states a fact, 0 if none does."""
    if not question.evidence:
        raise ValueError(f"{question.id} has no facts: its MRR is undefined")

    quotes = [quote.quote for evidence in question.evidence for quote in evidence.any_of]
    for rank, chunk in enumerate(chunks[:k], start=1):
        text = normalize(chunk.text)
        if any(contains(text, quote) for quote in quotes):
            return 1 / rank
    return 0
