"""Composition root: the one place that picks the implementations, from the settings.

Another store, embedding model or LLM is a new class, and a change here only.
"""

from openai import OpenAI
from sqlalchemy import create_engine

from agentic_rag.config import Settings
from agentic_rag.embeddings.base import Embedder
from agentic_rag.embeddings.openai import OpenAIEmbedder
from agentic_rag.storage.postgres import PostgresStore, database_url


def make_store(settings: Settings) -> PostgresStore:
    return PostgresStore(create_engine(database_url(settings.postgres)))


def make_embedder(settings: Settings) -> Embedder:
    openai = settings.openai
    client = OpenAI(api_key=openai.api_key.get_secret_value())
    return OpenAIEmbedder(client, openai.embedding_model, openai.embedding_dimensions)
