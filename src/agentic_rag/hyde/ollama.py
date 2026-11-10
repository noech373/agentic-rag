"""Hypothetical answers written by a model served by Ollama."""

import httpx

from agentic_rag.generation.ollama import chat
from agentic_rag.hyde.base import AnswerWriter
from agentic_rag.hyde.prompt import INSTRUCTIONS


class OllamaAnswerWriter(AnswerWriter):
    def __init__(self, client: httpx.Client, model: str, num_ctx: int) -> None:
        self._client = client
        self._model = model
        self._num_ctx = num_ctx

    def write(self, question: str) -> str:
        return chat(self._client, self._model, self._num_ctx, INSTRUCTIONS, question)
