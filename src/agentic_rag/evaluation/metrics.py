from collections.abc import Sequence

from agentic_rag.document import Chunk
from agentic_rag.evaluation.golden import Question, Quote, contains, normalize


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


def judged_at_k(
    question: Question, rejected: Sequence[Quote], chunks: Sequence[Chunk], k: int
) -> float:
    """The share of the first `k` chunks judged, useful or rejected: 1 if no chunk was retrieved.

    Divided by the chunks retrieved, not by `k`: a missing chunk is not an unjudged one.
    """
    if not question.evidence:
        raise ValueError(f"{question.id} has no facts: its judged rate is undefined")

    top = chunks[:k]
    if not top:
        return 1

    judged = sum(is_judged(question, rejected, chunk) for chunk in top)
    return judged / len(top)


def is_judged(question: Question, rejected: Sequence[Quote], chunk: Chunk) -> bool:
    """Whether the chunk states one of the question's facts or contains one of its rejections."""
    quotes = [quote.quote for evidence in question.evidence for quote in evidence.any_of]
    quotes += [quote.quote for quote in rejected]
    text = normalize(chunk.text)
    return any(contains(text, quote) for quote in quotes)
