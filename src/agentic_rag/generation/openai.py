"""OpenAI chat models, through the Responses API of the official SDK."""

from collections.abc import Sequence

from openai import OpenAI

from agentic_rag.generation.base import Answer, GenerationError, Generator
from agentic_rag.generation.prompt import INSTRUCTIONS, build_input
from agentic_rag.storage.base import RetrievedChunk


class OpenAIGenerator(Generator):
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def generate(self, question: str, sources: Sequence[RetrievedChunk]) -> Answer:
        response = self._client.responses.create(
            model=self._model,
            instructions=INSTRUCTIONS,
            input=build_input(question, sources),
            # By default OpenAI keeps each response for 30 days: questions may be confidential.
            store=False,
        )
        if response.status != "completed":  # e.g. out of tokens: the text would be cut short
            raise GenerationError(f"response {response.status}: {response.incomplete_details}")
        return Answer(text=response.output_text, sources=tuple(sources))
