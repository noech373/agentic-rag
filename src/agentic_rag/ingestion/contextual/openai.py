"""OpenAI chat models, through the Responses API of the official SDK."""

from openai import OpenAI

from agentic_rag.document import Chunk
from agentic_rag.generation.base import GenerationError
from agentic_rag.ingestion.contextual.base import Contextualizer
from agentic_rag.ingestion.contextual.prompt import INSTRUCTIONS, build_input


class OpenAIContextualizer(Contextualizer):
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def name(self) -> str:
        return self._model

    def situate(self, document: str, chunk: Chunk) -> str:
        response = self._client.responses.create(
            model=self._model,
            instructions=INSTRUCTIONS,
            input=build_input(document, chunk),
            store=False,
        )
        if response.status != "completed":  # e.g. out of tokens: the context would be cut short
            raise GenerationError(f"response {response.status}: {response.incomplete_details}")
        # An empty context would be embedded silently, as if the chunk had been situated
        if not response.output_text.strip():
            raise GenerationError("response without text")
        return response.output_text
