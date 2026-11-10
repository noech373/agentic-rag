"""Contexts written by a model served by Ollama."""

import httpx

from agentic_rag.document import Chunk
from agentic_rag.generation.ollama import chat
from agentic_rag.ingestion.contextual.base import Contextualizer
from agentic_rag.ingestion.contextual.prompt import INSTRUCTIONS, build_input


class OllamaContextualizer(Contextualizer):
    def __init__(self, client: httpx.Client, model: str, num_ctx: int) -> None:
        self._client = client
        self._model = model
        self._num_ctx = num_ctx  # the one the model is loaded with: another one reloads it

    @property
    def name(self) -> str:
        return self._model

    def situate(self, document: str, chunk: Chunk) -> str:
        return chat(
            self._client, self._model, self._num_ctx, INSTRUCTIONS, build_input(document, chunk)
        )
