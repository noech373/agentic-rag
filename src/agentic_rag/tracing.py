"""Tracing with Langfuse: each question becomes a trace of its retrieval and generation.

Optional: the factory wraps the RAG and its retriever with these classes only when the Langfuse
keys are set. The OpenAI calls themselves (embedding, generation, with tokens and cost) are
traced by Langfuse's OpenAI integration, nested under these observations.
"""

from langfuse import Langfuse

from agentic_rag.generation.base import Answer, Generator
from agentic_rag.generation.prompt import citation
from agentic_rag.rag import RAG
from agentic_rag.retrieval.base import Retriever
from agentic_rag.storage.base import RetrievedChunk


class TracedRAG(RAG):
    """One trace per question: the question in, the answer and what it cites out."""

    def __init__(
        self, retriever: Retriever, generator: Generator, langfuse: Langfuse, *, k: int = 8
    ) -> None:
        super().__init__(retriever, generator, k=k)
        self._langfuse = langfuse

    def ask(self, question: str) -> Answer:
        with self._langfuse.start_as_current_observation(
            name="ask", as_type="chain", input=question, metadata={"k": self._k}
        ) as observation:
            answer = super().ask(question)
            cited = {number: citation(source) for number, source in answer.cited.items()}
            observation.update(output={"text": answer.text, "cited": cited})
        return answer


class TracedRetriever(Retriever):
    """What the retriever found, with its scores: tells a retrieval miss from an LLM mistake."""

    def __init__(self, retriever: Retriever, langfuse: Langfuse) -> None:
        self._retriever = retriever
        self._langfuse = langfuse

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        with self._langfuse.start_as_current_observation(
            name="retrieve", as_type="retriever", input={"query": query, "k": k}
        ) as observation:
            chunks = self._retriever.retrieve(query, k)
            observation.update(output=[_summary(chunk) for chunk in chunks])
        return chunks


def _summary(chunk: RetrievedChunk) -> dict[str, object]:
    return {"citation": citation(chunk), "score": round(chunk.score, 4), "text": chunk.chunk.text}
