# agentic-rag

[![CI](https://github.com/noech373/agentic-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/noech373/agentic-rag/actions/workflows/ci.yml)

An agentic RAG (retrieval-augmented generation) assistant for EU regulatory compliance
(GDPR, AI Act, DORA, NIS2, ...), built with production practices in mind.

> **Status:** work in progress. A baseline RAG works end to end (ingestion, vector search,
> cited answers); the evaluation comes next, then measured improvements.

## Goals

- **Grounded answers with citations** down to the article level (e.g. *GDPR Art. 33(1)*),
  and an explicit "I don't know" when the sources do not cover the question.
- **Evaluation-driven development**: every retrieval or generation change is measured
  against a versioned test set before it is kept.
- **Swappable components**: the parsers, embedding model, store, retriever and LLM each sit
  behind an interface; adding one is a new class and one line in the factory.
- **Document-level access control**: users only retrieve documents they are allowed to see.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Project scaffolding: tooling, CI, local infrastructure, configuration | Done |
| 1 | Ingestion: layout-aware parsing (PDF tables, HTML, Markdown), structure-aware chunking, embeddings, incremental updates | Done |
| 2 | Baseline RAG: vector search, cited answers, "I don't know", tracing | In progress |
| 3 | Evaluation: golden dataset, retrieval and generation metrics, CI quality gates | Planned |
| 4 | Measured improvements: hybrid search (BM25 + vectors), reranking, contextual retrieval | Planned |
| 5 | Guardrails: input and output checks, prompt-injection defense | Planned |
| 6 | Production readiness: streaming API, user feedback, resilience, access control, more document sources | Planned |
| 7 | Agentic RAG with LangGraph: query routing, question decomposition, self-correction | Planned |

## Tech stack

Python 3.14, uv, PostgreSQL + pgvector + pg_textsearch (BM25), Docling, OpenAI (embeddings and
LLM). Planned: Langfuse (tracing), LangGraph (agentic phase).

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/) and [Docker](https://docs.docker.com/engine/install/) with the Compose and Buildx plugins.

```bash
uv sync                          # create .venv and install the locked dependencies
uv run pre-commit install        # run the checks automatically on every commit
cp .env.example .env             # then fill in your own values
docker compose up -d             # build and start PostgreSQL and Adminer

uv run agentic-rag download      # fetch the corpus (see below)
uv run agentic-rag convert       # run Docling on the PDFs once (~13 min on CPU, cached)
uv run agentic-rag ingest        # parse, chunk, embed and store it (~20 s; unchanged files are skipped)
uv run agentic-rag ask "Within how many hours must a personal data breach be notified?"
```

```text
A controller must notify the supervisory authority without undue delay and, where feasible,
no later than 72 hours after becoming aware of the breach. [3]

[3] GDPR, Article 33 https://data.europa.eu/eli/reg/2016/679/oj
```

`OPENAI_CHAT_MODEL` in `.env` picks the LLM: any model of the OpenAI Responses API.
`ask --retrieved` also lists the sources that were retrieved but not cited, with their scores.

Browse the database with Adminer at <http://localhost:8080/?pgsql=postgres&username=rag&db=rag>
(adjust `username` and `db` to your `.env`, then enter `POSTGRES_PASSWORD`).

`docker compose down` stops the containers and keeps the data; add `-v` to delete the data too.

## Architecture

```text
ingestion   manifest > download > parse (by format) > chunk > embed > store
question    embed > vector search (top k) > prompt with numbered sources > LLM > answer citing [n]
```

```text
src/agentic_rag/
├── document.py     domain types: Division, Block, Chunk
├── ingestion/      corpus download, parsers/ (EUR-Lex, PDF, Markdown), chunking, pipeline
├── embeddings/     base.py (Embedder), openai.py
├── storage/        base.py (ChunkStore, VectorSearch), postgres.py
├── retrieval/      base.py (Retriever), vector.py
├── generation/     base.py (Generator, Answer), prompt.py, openai.py
├── rag.py          retrieve, then generate
├── factory.py      the one place that picks the implementations, from the settings
└── cli.py, config.py
```

Each replaceable part has a package: `base.py` holds its interface, and each implementation is a
module next to it. A BM25 retriever, another embedding model or another database is a new module and
one line in `factory.py`; the rest of the code only knows the interfaces. The tests mirror this tree.

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
- **Markdown** (CNIL guide, internal policies): the headings give the path, read line by line.
  The lead paragraph that the CNIL styles as a heading under a sheet's title is text, and links
  keep only their text.

```bash
uv run agentic-rag convert    # run Docling on the PDFs once, cached in data/cache/ (~10 min)
```

## Chunking

Blocks follow the documents, so their sizes vary widely: a quarter of the corpus's 3282 blocks
have fewer than 50 tokens (GDPR Art. 33(2) has 20), while the AI Act definitions run to 3392.
The chunker evens them out without crossing a legal boundary:

- **Merge** consecutive blocks of the same article, recital, annex or point (or of the same
  heading, in guidelines) up to 512 tokens, counted with `cl100k_base`, the tokenizer of the
  embedding model.
- **Split** a longer block between its lines (subparagraphs, points, list items); a nested point
  stays with the line that introduces it.
- **Never split or merge a table**.

A chunk keeps the path its blocks share, and the numbers stay at the start of each line: GDPR
Article 33 becomes a single 319-token chunk whose lines start with "1." to "5.", so an answer can
still cite *Art. 33(1)*. A heading that a merged chunk drops from its path (the *GUIDANCE* and
*EXAMPLES OF EVIDENCE* parts of each ENISA point) becomes a line of its text. The corpus gives
1922 chunks, and a test checks that their text is exactly the text of the blocks, apart from these
heading lines. The 512-token budget is a starting point, to be compared with others in Phase 4.

Known limitations:

- A single line over the budget stays whole: a recital is one line, up to 833 tokens, well below
  the 8191 tokens the embedding model accepts.
- In PDFs, nesting only shows in the bullets, so a split can separate "...the following:" from
  its list (3 cases in the corpus).
- The pieces of a split PDF block all carry the page the block starts on.

## Retrieval and answers

- **Storage**: one PostgreSQL database for the documents, chunks and vectors. The vector column has
  no fixed dimension and each document records its embedding model, so another model needs no
  migration and a search only compares vectors of one model. Search is exact (cosine distance): on
  ~2000 chunks it takes ~9 ms, while an HNSW index would need tuning to find the true nearest chunks.
- **Ingestion is incremental**: a document is embedded again only when its file or the embedding
  model changes, and a file edited since its download is refused.
- **Answers**: the LLM gets the question and the 8 nearest chunks, each numbered and preceded by
  where it comes from (document, chapter, article and their titles, page). It must cite a number
  after each statement, answer only from the sources, and reply exactly *"I don't know based on the
  available sources."* otherwise. The CLI lists the cited sources with their links. OpenAI is asked
  not to keep the responses (`store=False`).

Known limitations, to be measured by the evaluation: vector search only (BM25 and reranking are
planned), a fixed number of sources, and no access control on the documents yet.

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
