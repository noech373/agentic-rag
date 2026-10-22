"""Question answering: retrieve the sources, then generate an answer that cites them."""

from agentic_rag.generation.base import NO_ANSWER, Answer, Generator
from agentic_rag.retrieval.base import Retriever


class RAG:
    def __init__(self, retriever: Retriever, generator: Generator, *, k: int = 8) -> None:
        self._retriever = retriever
        self._generator = generator
        self._k = k  # sources given to the LLM: ~8 x 250 tokens; to tune with the evaluation

    def ask(self, question: str) -> Answer:
        sources = self._retriever.retrieve(question, self._k)
        if not sources:  # nothing ingested: no need to ask the LLM
            return Answer(text=NO_ANSWER, sources=())
        return self._generator.generate(question, sources)
