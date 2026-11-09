"""Composition root: the one place that picks the implementations, from the settings and pipeline.

Another store, embedding model, retriever or LLM is a new class, and a change here only.
"""

import importlib

import httpx
from langfuse import Langfuse
from openai import OpenAI
from sqlalchemy import create_engine

from agentic_rag.config import LangfuseSettings, Settings
from agentic_rag.embeddings.base import Embedder
from agentic_rag.embeddings.openai import OpenAIEmbedder
from agentic_rag.evaluation.pooling.judge import OpenAIRelevanceJudge, RelevanceJudge
from agentic_rag.generation.openai import OpenAIGenerator
from agentic_rag.ingestion.contextual.base import Contextualizer
from agentic_rag.ingestion.contextual.cached import CONTEXTS_PATH, CachedContextualizer
from agentic_rag.ingestion.contextual.ollama import OllamaContextualizer
from agentic_rag.ingestion.contextual.openai import OpenAIContextualizer
from agentic_rag.ingestion.embedded_text import (
    ContextualTexts,
    TextsToEmbed,
    chunk_texts,
    titled_texts,
)
from agentic_rag.pipeline_config import ContextualizerName, PipelineConfig, RerankerName
from agentic_rag.rag import RAG
from agentic_rag.reranking.base import Reranker
from agentic_rag.reranking.retriever import RerankingRetriever
from agentic_rag.retrieval.base import Retriever
from agentic_rag.retrieval.bm25 import BM25Retriever
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.vector import VectorRetriever
from agentic_rag.storage.postgres import PostgresStore, database_url
from agentic_rag.tracing import TracedRAG, TracedRetriever


def make_store(settings: Settings, corpus: str = "public") -> PostgresStore:
    return PostgresStore(create_engine(database_url(settings.postgres)), corpus)


def make_embedder(settings: Settings) -> Embedder:
    openai = settings.openai
    return OpenAIEmbedder(_openai(settings), openai.embedding_model, openai.embedding_dimensions)


def make_texts_to_embed(settings: Settings, pipeline: PipelineConfig) -> TextsToEmbed:
    match pipeline.embedded_text:
        case "chunk":
            return chunk_texts
        case "titles":
            return titled_texts
        case "context":
            llm = _make_contextualizer(settings, pipeline.contextualizer)
            return ContextualTexts(CachedContextualizer(llm, CONTEXTS_PATH))


_OLLAMA_TIMEOUT = httpx.Timeout(1800, connect=10)


def _make_contextualizer(settings: Settings, name: ContextualizerName) -> Contextualizer:
    match name:
        case "openai":
            return OpenAIContextualizer(_openai(settings), settings.openai.chat_model)
        case "ollama":
            ollama = settings.ollama
            if ollama.base_url is None or ollama.model is None or ollama.num_ctx is None:
                raise ValueError(
                    "Ollama contexts need OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX"
                )
            client = httpx.Client(base_url=ollama.base_url, timeout=_OLLAMA_TIMEOUT)
            return OllamaContextualizer(client, ollama.model, ollama.num_ctx)


def make_retriever(settings: Settings, pipeline: PipelineConfig) -> Retriever:
    retriever = _make_first_stage_retriever(settings, pipeline)
    if pipeline.reranker is None:
        return retriever
    return RerankingRetriever(retriever, make_reranker(pipeline.reranker))


def make_reranker(name: RerankerName) -> Reranker:
    # Imported here: sentence-transformers takes 3 s to import, and only a reranker needs it
    from agentic_rag.reranking.cross_encoder import CrossEncoderReranker

    match name:
        case "bge-reranker-v2-m3":
            return CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
        case "ms-marco-minilm":
            return CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _make_first_stage_retriever(settings: Settings, pipeline: PipelineConfig) -> Retriever:
    match pipeline.retriever:
        case "vector":
            return VectorRetriever(make_store(settings, pipeline.corpus), make_embedder(settings))
        case "bm25":
            return BM25Retriever(make_store(settings, pipeline.corpus))
        case "hybrid":
            store = make_store(settings, pipeline.corpus)
            return HybridRetriever(
                [VectorRetriever(store, make_embedder(settings)), BM25Retriever(store)]
            )


def make_rag(settings: Settings, pipeline: PipelineConfig, *, k: int = 8) -> RAG:
    retriever = make_retriever(settings, pipeline)
    generator = OpenAIGenerator(_openai(settings), settings.openai.chat_model)
    if not settings.langfuse.enabled:
        return RAG(retriever, generator, k=k)
    langfuse = _langfuse(settings.langfuse)
    return TracedRAG(TracedRetriever(retriever, langfuse), generator, langfuse, k=k)


def make_relevance_judge(settings: Settings) -> RelevanceJudge:
    return OpenAIRelevanceJudge(_openai(settings), settings.openai.chat_model)


def _openai(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai.api_key.get_secret_value())


def _langfuse(settings: LangfuseSettings) -> Langfuse:
    # Importing it patches the OpenAI SDK, for the whole process: every call is then traced.
    importlib.import_module("langfuse.openai")
    secret_key = settings.secret_key.get_secret_value() if settings.secret_key else None
    return Langfuse(
        public_key=settings.public_key, secret_key=secret_key, base_url=settings.base_url
    )
