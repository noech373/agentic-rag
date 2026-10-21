# agentic-rag

[![CI](https://github.com/noech373/agentic-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/noech373/agentic-rag/actions/workflows/ci.yml)

An agentic RAG (retrieval-augmented generation) assistant for EU regulatory compliance
(GDPR, AI Act, DORA, NIS2, ...), built with production practices in mind.

> **Status:** work in progress, Phase 1 (ingestion).

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
| 0 | Project scaffolding: tooling, CI, local infrastructure, configuration | Done |
| 1 | Ingestion: pluggable connectors, layout-aware parsing (PDF tables, HTML, Markdown), structure-aware chunking, incremental updates, access control | In progress |
| 2 | Baseline RAG: vector search, cited answers, tracing | Planned |
| 3 | Evaluation: golden dataset, retrieval and generation metrics, CI quality gates | Planned |
| 4 | Measured improvements: hybrid search (BM25 + vectors), reranking, contextual retrieval | Planned |
| 5 | Guardrails: input and output checks, prompt-injection defense | Planned |
| 6 | Production readiness: streaming API, user feedback, resilience | Planned |
| 7 | Agentic RAG with LangGraph: query routing, question decomposition, self-correction | Planned |

## Tech stack

Python 3.14, uv, PostgreSQL + pgvector + pg_textsearch (BM25), LangChain, LangGraph, OpenAI, Langfuse.

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/) and [Docker](https://docs.docker.com/engine/install/) with the Compose and Buildx plugins.

```bash
uv sync                          # create .venv and install the locked dependencies
uv run pre-commit install        # run the checks automatically on every commit
cp .env.example .env             # then fill in your own values
docker compose up -d             # build and start PostgreSQL and Adminer
```

Browse the database with Adminer at <http://localhost:8080/?pgsql=postgres&username=rag&db=rag>
(adjust `username` and `db` to your `.env`, then enter `POSTGRES_PASSWORD`).

`docker compose down` stops the containers and keeps the data; add `-v` to delete the data too.

## Corpus

26 public documents on EU compliance, with a common thread (incident and breach notification
under GDPR, NIS2 and DORA):

- **EU legal acts** (XHTML): GDPR, AI Act, DORA, NIS2, Data Act.
- **Guidelines and reports** (PDF, with tables): EDPB Guidelines 9/2022 and 01/2021 on personal
  data breaches, ENISA NIS2 technical implementation guidance, ESAs final report on DORA
  incident reporting (JC 2024 33).
- **CNIL GDPR Developer Guide** (Markdown, 17 sheets).

```bash
uv run agentic-rag download   # fetch the corpus into data/corpus/ (~11 MB)
```

The documents are not stored in this repository. [`corpus/manifest.toml`](corpus/manifest.toml)
lists the download URL and the SHA-256 of each one: a file whose checksum differs is rejected, so
every checkout ingests exactly the same bytes and the evaluation stays valid. Re-running the
command only downloads what is missing or corrupted.

EU legal acts are fetched from [Cellar](https://op.europa.eu/en/web/cellar), the Publications
Office's repository for machine access. Each document keeps its own license: EU legal acts may be
reused under [Commission Decision 2011/833/EU](https://data.europa.eu/eli/dec/2011/833/oj), the
CNIL guide is published under [GPL-3.0](https://github.com/LINCnil/GDPR-Developer-Guide), and the
EDPB, ENISA and ESAs documents under their publishers' reuse terms.

## Parsing

Each document is split into blocks tagged with their place in the document, so that answers can
cite it: *GDPR, Article 33(1)*, *EDPB Guidelines 9/2022, II. > A. > 1. Article 33 requirements,
p. 10*.

- **EU legal acts**: a dedicated parser reads the structure from the identifiers of the
  Official Journal's XHTML (`art_33`, `033.001`).
- **PDFs**: [Docling](https://github.com/docling-project/docling) finds headings, lists and
  tables with a layout model, on CPU and without OCR (every PDF has a text layer), and nests the
  headings from the PDF bookmarks and the numbering. Two rules, measured on the corpus: headings
  that are sentences ("49. Advisable measures:") are turned back into text, as they would shift
  the chapters after them; the points of a legal text quoted by ENISA ("3.5.1. The relevant
  entities shall...") are divisions, with their guidance nested below. As for legal acts, a
  block is the content of one division (a table is a block of its own), footnotes are skipped,
  and each block keeps the page it starts on.

```bash
uv run agentic-rag convert    # run Docling on the PDFs once, cached in data/cache/ (~10 min)
```

## Development

```bash
uv run pytest -m "not integration and not eval"   # fast unit tests
uv run pre-commit run --all-files                 # every check the CI runs
uv run ruff check .                               # lint only
uv run ruff format .                              # format only
uv run pyright                                    # type-check only (strict mode)
```

## License

[MIT](LICENSE)
