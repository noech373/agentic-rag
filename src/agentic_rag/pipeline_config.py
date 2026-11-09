"""Which bricks the RAG is built from, read from a TOML file per variant (`pipelines/<name>.toml`).

The defaults are the baseline: an empty file, or no file at all, is the plain vector RAG.
"""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

RerankerName = Literal["bge-reranker-v2-m3", "ms-marco-minilm"]
# An ingested variant, a Postgres schema: its name ends up in SQL, hence the pattern
CorpusName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]


class PipelineConfig(BaseModel):
    # extra="forbid": a misspelled key would otherwise run the baseline under another name
    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus: CorpusName = "public"
    # Read by `ingest` only: the corpus holds the vectors of this text
    embedded_text: Literal["chunk", "titles"] = "chunk"
    retriever: Literal["vector", "bm25", "hybrid"] = "vector"
    reranker: RerankerName | None = None


def load_pipeline(path: Path) -> PipelineConfig:
    with path.open("rb") as f:
        return PipelineConfig.model_validate(tomllib.load(f))
