"""Composition root: the one place that picks the implementations, from the settings and pipeline.

Another store, embedding model, retriever or LLM is a new class, and a change here only.
"""

import importlib

from langfuse import Langfuse
from openai import OpenAI
from sqlalchemy import create_engine

from agentic_rag.config import LangfuseSettings, Settings
from agentic_rag.embeddings.base import Embedder
from agentic_rag.embeddings.openai import OpenAIEmbedder
from agentic_rag.generation.openai import OpenAIGenerator
from agentic_rag.pipeline_config import PipelineConfig
from agentic_rag.rag import RAG
from agentic_rag.retrieval.base import Retriever
from agentic_rag.retrieval.vector import VectorRetriever
from agentic_rag.storage.postgres import PostgresStore, database_url
from agentic_rag.tracing import TracedRAG, TracedRetriever


def make_store(settings: Settings) -> PostgresStore:
    return PostgresStore(create_engine(database_url(settings.postgres)))


def make_embedder(settings: Settings) -> Embedder:
    openai = settings.openai
    return OpenAIEmbedder(_openai(settings), openai.embedding_model, openai.embedding_dimensions)


def make_retriever(settings: Settings, pipeline: PipelineConfig) -> Retriever:
    match pipeline.retriever:
        case "vector":
            return VectorRetriever(make_store(settings), make_embedder(settings))


def make_rag(settings: Settings, pipeline: PipelineConfig, *, k: int = 8) -> RAG:
    retriever = make_retriever(settings, pipeline)
    generator = OpenAIGenerator(_openai(settings), settings.openai.chat_model)
    if not settings.langfuse.enabled:
        return RAG(retriever, generator, k=k)
    langfuse = _langfuse(settings.langfuse)
    return TracedRAG(TracedRetriever(retriever, langfuse), generator, langfuse, k=k)


def _openai(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai.api_key.get_secret_value())


def _langfuse(settings: LangfuseSettings) -> Langfuse:
    # Importing it patches the OpenAI SDK, for the whole process: every call is then traced.
    importlib.import_module("langfuse.openai")
    secret_key = settings.secret_key.get_secret_value() if settings.secret_key else None
    return Langfuse(
        public_key=settings.public_key, secret_key=secret_key, base_url=settings.base_url
    )
