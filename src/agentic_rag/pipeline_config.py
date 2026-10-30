"""Which bricks the RAG is built from, read from a TOML file per variant (`pipelines/<name>.toml`).

The defaults are the baseline: an empty file, or no file at all, is the plain vector RAG.
"""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class PipelineConfig(BaseModel):
    # extra="forbid": a misspelled key would otherwise run the baseline under another name
    model_config = ConfigDict(frozen=True, extra="forbid")

    retriever: Literal["vector", "bm25", "hybrid"] = "vector"


def load_pipeline(path: Path) -> PipelineConfig:
    with path.open("rb") as f:
        return PipelineConfig.model_validate(tomllib.load(f))
