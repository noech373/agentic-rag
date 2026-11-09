"""Models served by Ollama, through its native chat API.

Not the OpenAI-compatible one: only the native API sets the context length per request, and
tells how many prompt tokens were read, which shows a truncated document.
"""

import httpx

from agentic_rag.document import Chunk
from agentic_rag.generation.base import GenerationError
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
        response = self._client.post(
            "/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "think": False,
                "options": {"num_ctx": self._num_ctx, "temperature": 0},
                "messages": [
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": build_input(document, chunk)},
                ],
            },
        )
        response.raise_for_status()
        body = response.json()
        if body.get("done_reason") != "stop":
            raise GenerationError(f"response {body.get('done_reason')}")
        # Ollama cuts a prompt too long for the window without an error: the document is partial
        if body["prompt_eval_count"] + body["eval_count"] >= self._num_ctx:
            raise GenerationError(f"prompt of {body['prompt_eval_count']} tokens, maybe truncated")
        text: str = body["message"]["content"]
        if not text.strip():
            raise GenerationError("response without text")
        return text
