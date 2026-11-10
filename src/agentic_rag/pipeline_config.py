"""Which bricks the RAG is built from, read from a TOML file per variant (`pipelines/<name>.toml`).

The defaults are the best composition adopted so far: an empty file, or no file at all.
`pipelines/baseline.toml` keeps the plain RAG, the start of the ablation.
"""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

EmbeddedText = Literal["chunk", "titles", "context"]
LLMName = Literal["openai", "ollama"]  # the model behind each is set in .env
RerankerName = Literal["bge-reranker-v2-m3", "ms-marco-minilm"]
# An ingested variant, a Postgres schema: its name ends up in SQL, hence the pattern
CorpusName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]


class PipelineConfig(BaseModel):
    # extra="forbid": a misspelled key would otherwise run the defaults under another name
    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus: CorpusName = "titles"
    embedded_text: EmbeddedText = "titles"
    contextualizer: LLMName = "openai"
    retriever: Literal["vector", "bm25", "hybrid"] = "vector"
    hyde: Literal["ollama"] | None = None
    reranker: RerankerName | None = None
    generator: LLMName = "openai"


def load_pipeline(path: Path) -> PipelineConfig:
    with path.open("rb") as f:
        return PipelineConfig.model_validate(tomllib.load(f))
