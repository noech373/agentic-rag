# agentic-rag

An agentic RAG (retrieval-augmented generation) assistant for EU regulatory compliance
(GDPR, AI Act, DORA, NIS2, ...), built with production practices in mind.

> **Status:** work in progress, Phase 0 (project scaffolding).

## Goals

- **Grounded answers with citations** down to the article level (e.g. *GDPR Art. 33(1)*),
  and an explicit "I don't know" when the sources do not cover the question.
- **Evaluation-driven development**: every retrieval or generation change is measured
  against a versioned test set before it is kept.
- **Extensible ingestion**: adding a new document source means writing one connector class.
- **Document-level access control**: users only retrieve documents they are allowed to see.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Project scaffolding: tooling, CI, local infrastructure, configuration | In progress |
| 1 | Ingestion: pluggable connectors, layout-aware parsing (PDF tables, HTML, Markdown), structure-aware chunking, incremental updates, access control | Planned |
| 2 | Baseline RAG: vector search, cited answers, tracing | Planned |
| 3 | Evaluation: golden dataset, retrieval and generation metrics, CI quality gates | Planned |
| 4 | Measured improvements: hybrid search (BM25 + vectors), reranking, contextual retrieval | Planned |
| 5 | Guardrails: input and output checks, prompt-injection defense | Planned |
| 6 | Production readiness: streaming API, user feedback, resilience | Planned |
| 7 | Agentic RAG with LangGraph: query routing, question decomposition, self-correction | Planned |

## Tech stack

Python 3.14, uv, PostgreSQL + pgvector, LangChain, LangGraph, OpenAI, Langfuse.

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/).

```bash
uv sync    # create .venv and install the locked dependencies
```

## Development

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type-check (strict mode)
```

## License

[MIT](LICENSE)
