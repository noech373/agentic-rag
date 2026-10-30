"""Judgments: the quotes of retrieved chunks judged not to stateany fact of a question."""

import tomllib
from pathlib import Path

from pydantic import TypeAdapter

from agentic_rag.evaluation.golden import Quote

type Judgments = dict[str, tuple[Quote, ...]]  # question id → its rejected quotes


def load_judgments(path: Path) -> Judgments:
    with path.open("rb") as f:
        return TypeAdapter(Judgments).validate_python(tomllib.load(f))
