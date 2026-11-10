"""Models served by Ollama, through its native chat API.

Not the OpenAI-compatible one: only the native API sets the context length per request, and
tells how many prompt tokens were read, which shows a truncated prompt.
"""

from collections.abc import Sequence

import httpx

from agentic_rag.document import RetrievedChunk
from agentic_rag.generation.base import Answer, GenerationError, Generator
from agentic_rag.generation.prompt import INSTRUCTIONS, build_input


class OllamaGenerator(Generator):
    def __init__(self, client: httpx.Client, model: str, num_ctx: int) -> None:
        self._client = client
        self._model = model
        self._num_ctx = num_ctx  # the one the model is loaded with: another one reloads it

    def generate(self, question: str, sources: Sequence[RetrievedChunk]) -> Answer:
        text = chat(
            self._client,
            self._model,
            self._num_ctx,
            INSTRUCTIONS,
            build_input(question, sources),
        )
        return Answer(text=text, sources=tuple(sources))


def chat(client: httpx.Client, model: str, num_ctx: int, instructions: str, text: str) -> str:
    """The model's answer to `text`, the most likely words, without thinking."""
    response = client.post(
        "/api/chat",
        json={
            "model": model,
            "stream": False,
            "think": False,
            "options": {"num_ctx": num_ctx, "temperature": 0},
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": text},
            ],
        },
    )
    response.raise_for_status()
    body = response.json()
    if body.get("done_reason") != "stop":
        raise GenerationError(f"response {body.get('done_reason')}")
    # Ollama cuts a prompt too long for the window without an error
    if body["prompt_eval_count"] + body["eval_count"] >= num_ctx:
        raise GenerationError(f"prompt of {body['prompt_eval_count']} tokens, maybe truncated")
    answer: str = body["message"]["content"]
    if not answer.strip():
        raise GenerationError("response without text")
    return answer
