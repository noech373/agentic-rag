"""Embedding models: texts to vectors, for the vector search.

The rest of the code depends only on `Embedder`: another model or provider is one new class.
"""

from collections.abc import Iterator, Sequence
from typing import Protocol

from openai import OpenAI


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """The vector space, stored with the vectors: "text-embedding-3-small@1536"."""
        ...

    # Two methods, as in LangChain: some models (e5, Cohere, Voyage) embed a query differently
    # from the documents it should match.
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# Limits of the OpenAI embeddings endpoint, per request.
_MAX_INPUTS = 2048
_MAX_TOKENS = 300_000  # summed over the inputs


class OpenAIEmbedder(Embedder):
    def __init__(self, client: OpenAI, model: str, dimensions: int) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"{self._model}@{self._dimensions}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batches(texts):
            response = self._client.embeddings.create(
                model=self._model, input=list(batch), dimensions=self._dimensions
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _batches(texts: Sequence[str]) -> Iterator[Sequence[str]]:
    """Consecutive texts, as many per request as the API accepts.

    A token is at least one byte of UTF-8: counting bytes stays under the token limit without
    the model's tokenizer. A single text over the model's own limit fails at the API.
    """
    start, size = 0, 0
    for end, text in enumerate(texts):
        length = len(text.encode())
        if end > start and (end - start == _MAX_INPUTS or size + length > _MAX_TOKENS):
            yield texts[start:end]
            start, size = end, 0
        size += length
    if start < len(texts):
        yield texts[start:]
