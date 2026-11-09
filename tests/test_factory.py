import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
from pydantic import SecretStr

from agentic_rag import factory
from agentic_rag.config import LangfuseSettings, OpenAISettings, PostgresSettings, Settings
from agentic_rag.document import Chunk, Division
from agentic_rag.factory import make_rag, make_retriever, make_texts_to_embed
from agentic_rag.ingestion.contextual.cached import CONTEXTS_PATH, CachedContextualizer
from agentic_rag.ingestion.contextual.prompt import render_document
from agentic_rag.ingestion.embedded_text import ContextualTexts, chunk_texts, titled_texts
from agentic_rag.pipeline_config import PipelineConfig, RerankerName
from agentic_rag.rag import RAG
from agentic_rag.reranking import cross_encoder
from agentic_rag.reranking.retriever import RerankingRetriever
from agentic_rag.retrieval.bm25 import BM25Retriever
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.vector import VectorRetriever
from agentic_rag.storage.postgres import PostgresStore


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Fake values: building the clients connects to nothing, so no network or database."""
    monkeypatch.chdir(tmp_path)  # no .env file in this empty directory
    return Settings(
        openai=OpenAISettings(api_key=SecretStr("sk-test"), chat_model="some-chat-model"),
        postgres=PostgresSettings(user="u", password=SecretStr("p"), db="d"),
        langfuse=LangfuseSettings(public_key=None, secret_key=None),
    )


def test_a_store_is_on_the_public_corpus_unless_told_otherwise(settings: Settings) -> None:
    assert factory.make_store(settings).corpus == "public"
    assert factory.make_store(settings, "titles").corpus == "titles"


def _corpus(retriever: VectorRetriever | BM25Retriever) -> str:
    store = retriever._search  # pyright: ignore[reportPrivateUsage]
    assert isinstance(store, PostgresStore)
    return store.corpus


def test_the_vector_retriever_searches_the_corpus_of_its_pipeline(settings: Settings) -> None:
    retriever = make_retriever(settings, PipelineConfig(retriever="vector", corpus="titles"))

    assert isinstance(retriever, VectorRetriever)
    assert _corpus(retriever) == "titles"


def test_the_bm25_retriever_searches_the_corpus_of_its_pipeline(settings: Settings) -> None:
    retriever = make_retriever(settings, PipelineConfig(retriever="bm25", corpus="titles"))

    assert isinstance(retriever, BM25Retriever)
    assert _corpus(retriever) == "titles"


def test_the_hybrid_retriever_searches_the_corpus_of_its_pipeline(settings: Settings) -> None:
    retriever = make_retriever(settings, PipelineConfig(retriever="hybrid", corpus="titles"))

    assert isinstance(retriever, HybridRetriever)
    vector, bm25 = retriever._retrievers  # pyright: ignore[reportPrivateUsage]
    assert isinstance(vector, VectorRetriever)
    assert isinstance(bm25, BM25Retriever)
    assert (_corpus(vector), _corpus(bm25)) == ("titles", "titles")


def test_the_vector_pipeline_builds_a_vector_retriever(settings: Settings) -> None:
    retriever = make_retriever(settings, PipelineConfig(retriever="vector"))

    assert isinstance(retriever, VectorRetriever)


def test_the_bm25_pipeline_builds_a_bm25_retriever_without_openai(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_embedder(settings: Settings) -> None:
        raise AssertionError("BM25 needs no embedding")

    monkeypatch.setattr(factory, "make_embedder", no_embedder)

    assert isinstance(make_retriever(settings, PipelineConfig(retriever="bm25")), BM25Retriever)


def test_the_hybrid_pipeline_fuses_vector_then_bm25_on_one_store(settings: Settings) -> None:
    """Vector first: it wins the ties of the fusion, being the better retriever alone."""
    retriever = make_retriever(settings, PipelineConfig(retriever="hybrid"))

    assert isinstance(retriever, HybridRetriever)
    vector, bm25 = retriever._retrievers  # pyright: ignore[reportPrivateUsage]
    assert isinstance(vector, VectorRetriever)
    assert isinstance(bm25, BM25Retriever)
    assert vector._search is bm25._search  # pyright: ignore[reportPrivateUsage]


def test_the_rag_gets_the_retriever_of_its_pipeline(settings: Settings) -> None:
    rag = make_rag(settings, PipelineConfig(retriever="vector"))

    assert type(rag) is RAG  # tracing is off without Langfuse keys
    assert isinstance(rag._retriever, VectorRetriever)  # pyright: ignore[reportPrivateUsage]


class _FakeCrossEncoder:
    """Stands for CrossEncoderReranker: no model is loaded in a unit test."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


@pytest.fixture
def fake_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cross_encoder, "CrossEncoderReranker", _FakeCrossEncoder)


def test_without_a_reranker_the_retriever_is_not_wrapped(settings: Settings) -> None:
    retriever = make_retriever(settings, PipelineConfig(reranker=None))

    assert isinstance(retriever, VectorRetriever)


@pytest.mark.parametrize(
    ("name", "model_id"),
    [
        ("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3"),
        ("ms-marco-minilm", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ],
)
@pytest.mark.usefixtures("fake_cross_encoder")
def test_a_reranker_name_loads_its_model(
    settings: Settings, name: RerankerName, model_id: str
) -> None:
    retriever = make_retriever(settings, PipelineConfig(reranker=name))

    assert isinstance(retriever, RerankingRetriever)
    reranker = retriever._reranker  # pyright: ignore[reportPrivateUsage]
    assert isinstance(reranker, _FakeCrossEncoder)
    assert reranker.model_name == model_id


@pytest.mark.parametrize("first", ["vector", "bm25", "hybrid"])
@pytest.mark.usefixtures("fake_cross_encoder")
def test_the_reranker_wraps_whichever_retriever_is_chosen(
    settings: Settings, first: Literal["vector", "bm25", "hybrid"]
) -> None:
    pipeline = PipelineConfig(retriever=first, reranker="ms-marco-minilm")

    retriever = make_retriever(settings, pipeline)

    assert isinstance(retriever, RerankingRetriever)
    expected = {"vector": VectorRetriever, "bm25": BM25Retriever, "hybrid": HybridRetriever}[first]
    assert isinstance(retriever._retriever, expected)  # pyright: ignore[reportPrivateUsage]


def test_a_pipeline_without_a_reranker_never_imports_the_model_library() -> None:
    """sentence-transformers takes 3 s to import: only a reranker pays for it."""
    code = (
        "import sys\n"
        "from agentic_rag import factory\n"
        "assert 'sentence_transformers' not in sys.modules\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603 - our own code


def test_the_chunk_text_alone_is_embedded_by_default(settings: Settings) -> None:
    assert make_texts_to_embed(settings, PipelineConfig().embedded_text) is chunk_texts


def test_the_titles_mode_embeds_the_titles(settings: Settings) -> None:
    assert make_texts_to_embed(settings, "titles") is titled_texts


def test_the_context_mode_reads_the_committed_cache_before_asking_the_llm(
    settings: Settings,
) -> None:
    """With a fake API key, a call to OpenAI would fail: the context comes from the cache."""
    chunk = Chunk(kind="text", text="a", path=(Division(kind="article", number="33"),))

    class _Fake:
        name = settings.openai.chat_model

        def situate(self, document: str, chunk: Chunk) -> str:
            return "cached context"

    CONTEXTS_PATH.parent.mkdir()
    CachedContextualizer(_Fake(), CONTEXTS_PATH).situate(render_document("GDPR", [chunk]), chunk)

    texts_to_embed = make_texts_to_embed(settings, "context")

    assert isinstance(texts_to_embed, ContextualTexts)
    assert texts_to_embed("GDPR", [chunk]) == ["cached context\n\na"]
