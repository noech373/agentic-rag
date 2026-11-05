"""A local cross-encoder: the model reads each question and text together."""

from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from agentic_rag.reranking.base import Reranker


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str) -> None:
        self._model = CrossEncoder(model_name, device="cpu", max_length=512)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One score per text, in the same order; the scale depends on the model."""
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs, batch_size=8, show_progress_bar=False)  # pyright: ignore[reportUnknownMemberType]
        return scores.tolist()
