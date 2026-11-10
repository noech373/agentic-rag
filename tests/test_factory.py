import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
from pydantic import SecretStr

from agentic_rag import factory
from agentic_rag.config import (
    LangfuseSettings,
    OllamaSettings,
    OpenAISettings,
    PostgresSettings,
    Settings,
)
from agentic_rag.document import Chunk, Division
from agentic_rag.factory import make_rag, make_retriever, make_texts_to_embed
from agentic_rag.generation.ollama import OllamaGenerator
from agentic_rag.generation.openai import OpenAIGenerator
from agentic_rag.hyde.cached import ANSWERS_PATH, CachedAnswerWriter
from agentic_rag.hyde.ollama import OllamaAnswerWriter
from agentic_rag.hyde.retriever import HydeRetriever
from agentic_rag.ingestion.contextual.cached import CONTEXTS_PATH, CachedContextualizer
from agentic_rag.ingestion.contextual.ollama import OllamaContextualizer
from agentic_rag.ingestion.contextual.openai import OpenAIContextualizer
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
        ollama=OllamaSettings(base_url=None, model=None, num_ctx=None),
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


def test_the_chunk_mode_embeds_the_chunk_text_alone(settings: Settings) -> None:
    assert make_texts_to_embed(settings, PipelineConfig(embedded_text="chunk")) is chunk_texts


def test_the_titles_mode_embeds_the_titles(settings: Settings) -> None:
    assert make_texts_to_embed(settings, PipelineConfig(embedded_text="titles")) is titled_texts


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

    texts_to_embed = make_texts_to_embed(settings, PipelineConfig(embedded_text="context"))

    assert isinstance(texts_to_embed, ContextualTexts)
    assert texts_to_embed("GDPR", [chunk]) == ["cached context\n\na"]


def _contextualizer(settings: Settings, pipeline: PipelineConfig) -> object:
    texts_to_embed = make_texts_to_embed(settings, pipeline)
    assert isinstance(texts_to_embed, ContextualTexts)
    cached = texts_to_embed.contextualizer
    assert isinstance(cached, CachedContextualizer)
    return cached._contextualizer  # pyright: ignore[reportPrivateUsage]


def test_openai_writes_the_contexts_by_default(settings: Settings) -> None:
    llm = _contextualizer(settings, PipelineConfig(embedded_text="context"))

    assert isinstance(llm, OpenAIContextualizer)
    assert llm.name == "some-chat-model"


def test_ollama_writes_the_contexts_when_the_pipeline_asks_for_it(settings: Settings) -> None:
    ollama = OllamaSettings(base_url="https://ollama.test", model="qwen-test", num_ctx=1000)
    settings = settings.model_copy(update={"ollama": ollama})

    llm = _contextualizer(
        settings, PipelineConfig(embedded_text="context", contextualizer="ollama")
    )

    assert isinstance(llm, OllamaContextualizer)
    assert llm.name == "qwen-test"


def test_ollama_contexts_without_the_ollama_settings_fail_before_any_chunk(
    settings: Settings,
) -> None:
    with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
        make_texts_to_embed(
            settings, PipelineConfig(embedded_text="context", contextualizer="ollama")
        )


@pytest.fixture
def qwen(settings: Settings) -> Settings:
    ollama = OllamaSettings(base_url="https://ollama.test", model="qwen-test", num_ctx=1000)
    return settings.model_copy(update={"ollama": ollama})


def test_without_hyde_the_retriever_searches_with_the_question(qwen: Settings) -> None:
    assert isinstance(make_retriever(qwen, PipelineConfig()), VectorRetriever)


@pytest.mark.parametrize("first", ["vector", "bm25", "hybrid"])
def test_hyde_wraps_whichever_retriever_is_chosen(
    qwen: Settings, first: Literal["vector", "bm25", "hybrid"]
) -> None:
    retriever = make_retriever(qwen, PipelineConfig(retriever=first, hyde="ollama"))

    assert isinstance(retriever, HydeRetriever)
    expected = {"vector": VectorRetriever, "bm25": BM25Retriever, "hybrid": HybridRetriever}[first]
    assert isinstance(retriever._retriever, expected)  # pyright: ignore[reportPrivateUsage]


def test_qwen_writes_the_answers_through_the_committed_cache(qwen: Settings) -> None:
    retriever = make_retriever(qwen, PipelineConfig(hyde="ollama"))

    assert isinstance(retriever, HydeRetriever)
    cached = retriever._writer  # pyright: ignore[reportPrivateUsage]
    assert isinstance(cached, CachedAnswerWriter)
    assert cached._path == ANSWERS_PATH  # pyright: ignore[reportPrivateUsage]
    writer = cached._writer  # pyright: ignore[reportPrivateUsage]
    assert isinstance(writer, OllamaAnswerWriter)
    assert writer.name == "qwen-test"


def test_hyde_reads_the_committed_cache_before_asking_qwen(qwen: Settings) -> None:
    """ollama.test does not exist: the answer can only come from the cache."""

    class _Fake:
        name = "qwen-test"

        def write(self, question: str) -> str:
            return "cached answer"

    ANSWERS_PATH.parent.mkdir()
    CachedAnswerWriter(_Fake(), ANSWERS_PATH).write("Can we dismiss our DPO?")

    retriever = make_retriever(qwen, PipelineConfig(hyde="ollama"))

    assert isinstance(retriever, HydeRetriever)
    writer = retriever._writer  # pyright: ignore[reportPrivateUsage]
    assert writer.write("Can we dismiss our DPO?") == "cached answer"


@pytest.mark.usefixtures("fake_cross_encoder")
def test_the_reranker_scores_against_the_question_not_the_hyde_answer(qwen: Settings) -> None:
    retriever = make_retriever(qwen, PipelineConfig(hyde="ollama", reranker="ms-marco-minilm"))

    assert isinstance(retriever, RerankingRetriever)
    assert isinstance(retriever._retriever, HydeRetriever)  # pyright: ignore[reportPrivateUsage]


def test_hyde_without_the_ollama_settings_fails_before_any_question(settings: Settings) -> None:
    with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
        make_retriever(settings, PipelineConfig(hyde="ollama"))


def test_openai_writes_the_answers_by_default(settings: Settings) -> None:
    rag = make_rag(settings, PipelineConfig())

    assert isinstance(rag._generator, OpenAIGenerator)  # pyright: ignore[reportPrivateUsage]


def test_ollama_writes_the_answers_when_the_pipeline_asks_for_it(qwen: Settings) -> None:
    rag = make_rag(qwen, PipelineConfig(generator="ollama"))

    assert isinstance(rag._generator, OllamaGenerator)  # pyright: ignore[reportPrivateUsage]


def test_an_ollama_generator_without_the_ollama_settings_fails_before_any_question(
    settings: Settings,
) -> None:
    with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
        make_rag(settings, PipelineConfig(generator="ollama"))
