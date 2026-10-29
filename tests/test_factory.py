from pathlib import Path

import pytest
from pydantic import SecretStr

from agentic_rag.config import LangfuseSettings, OpenAISettings, PostgresSettings, Settings
from agentic_rag.factory import make_rag, make_retriever
from agentic_rag.pipeline_config import PipelineConfig
from agentic_rag.rag import RAG
from agentic_rag.retrieval.vector import VectorRetriever


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Fake values: building the clients connects to nothing, so no network or database."""
    monkeypatch.chdir(tmp_path)  # no .env file in this empty directory
    return Settings(
        openai=OpenAISettings(api_key=SecretStr("sk-test"), chat_model="some-chat-model"),
        postgres=PostgresSettings(user="u", password=SecretStr("p"), db="d"),
        langfuse=LangfuseSettings(public_key=None, secret_key=None),
    )


def test_the_vector_pipeline_builds_a_vector_retriever(settings: Settings) -> None:
    retriever = make_retriever(settings, PipelineConfig(retriever="vector"))

    assert isinstance(retriever, VectorRetriever)


def test_the_rag_gets_the_retriever_of_its_pipeline(settings: Settings) -> None:
    rag = make_rag(settings, PipelineConfig(retriever="vector"))

    assert type(rag) is RAG  # tracing is off without Langfuse keys
    assert isinstance(rag._retriever, VectorRetriever)  # pyright: ignore[reportPrivateUsage]
