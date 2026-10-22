"""Composition root: the one place that picks the implementations, from the settings.

Another store, embedding model, retriever or LLM is a new class, and a change here only.
"""

from openai import OpenAI
from sqlalchemy import create_engine

from agentic_rag.config import Settings
from agentic_rag.embeddings.base import Embedder
from agentic_rag.embeddings.openai import OpenAIEmbedder
from agentic_rag.generation.openai import OpenAIGenerator
from agentic_rag.rag import RAG
from agentic_rag.retrieval.vector import VectorRetriever
from agentic_rag.storage.postgres import PostgresStore, database_url


def make_store(settings: Settings) -> PostgresStore:
    return PostgresStore(create_engine(database_url(settings.postgres)))


def make_embedder(settings: Settings) -> Embedder:
    openai = settings.openai
    return OpenAIEmbedder(_openai(settings), openai.embedding_model, openai.embedding_dimensions)


def make_rag(settings: Settings, *, k: int = 8) -> RAG:
    retriever = VectorRetriever(make_store(settings), make_embedder(settings))
    generator = OpenAIGenerator(_openai(settings), settings.openai.chat_model)
    return RAG(retriever, generator, k=k)


def _openai(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai.api_key.get_secret_value())
