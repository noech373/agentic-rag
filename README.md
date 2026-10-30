# agentic-rag

[![CI](https://github.com/noech373/agentic-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/noech373/agentic-rag/actions/workflows/ci.yml)

An agentic RAG (retrieval-augmented generation) assistant for EU regulatory compliance
(GDPR, AI Act, DORA, NIS2, ...), built with production practices in mind.

> **Status:** work in progress. A baseline RAG works end to end (ingestion, vector search,
> cited answers, tracing), its retrieval is measured on a golden set of 104 questions, and the
> improvements are being measured one by one (see [Results and findings](#results-and-findings)).

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
| 2 | Baseline RAG: vector search, cited answers, "I don't know", tracing | Done |
| 3 | Evaluation: golden dataset, retrieval and generation metrics, CI quality gates | In progress |
| 4 | Measured improvements: hybrid search (BM25 + vectors), reranking, contextual retrieval | In progress |
| 5 | Guardrails: input and output checks, prompt-injection defense | Planned |
| 6 | Production readiness: streaming API, user feedback, resilience, access control, more document sources | Planned |
| 7 | Agentic RAG with LangGraph: query routing, question decomposition, self-correction | Planned |

## Tech stack

Python 3.14, uv, PostgreSQL + pgvector + pg_textsearch (BM25), Docling, OpenAI (embeddings and
LLM), Langfuse (tracing). Planned: LangGraph (agentic phase).

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
question    retrieve (vectors, BM25, or both fused) > prompt with numbered sources > LLM > answer citing [n]
```

```text
src/agentic_rag/
├── document.py     domain types: Division, Block, Chunk
├── ingestion/      corpus download, parsers/ (EUR-Lex, PDF, Markdown), chunking, pipeline
├── embeddings/     base.py (Embedder), openai.py
├── storage/        base.py (ChunkStore, VectorSearch, KeywordSearch), postgres.py
├── retrieval/      base.py (Retriever), vector.py, bm25.py, hybrid.py
├── generation/     base.py (Generator, Answer), prompt.py, openai.py
├── rag.py          retrieve, then generate
├── tracing.py      Langfuse wrappers around the RAG and its retriever
├── pipeline_config.py  which bricks a pipeline is built from, read from pipelines/<name>.toml
├── factory.py      the one place that picks the implementations, from the settings and the pipeline
└── cli.py, config.py
pipelines/          one TOML file per variant of the RAG (the defaults are the baseline)
```

Each replaceable part has a package: `base.py` holds its interface, and each implementation is a
module next to it. A reranker, another embedding model or another database is a new module and
one case in `factory.py`; the rest of the code only knows the interfaces. The tests mirror this tree.

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

The documents are not stored in this repository. [`corpus/manifest.toml`](corpus/manifest.toml) lists the download URL and the SHA-256 of each one: a file whose checksum differs is rejected, so every checkout ingests exactly the same bytes and the evaluation stays valid. Re-running the command only downloads what is missing or corrupted.

EU legal acts are fetched from [Cellar](https://op.europa.eu/en/web/cellar), the Publications Office's repository for machine access. Each document keeps its own license: EU legal acts may be reused under [Commission Decision 2011/833/EU](https://data.europa.eu/eli/dec/2011/833/oj), the CNIL guide is published under [GPL-3.0](https://github.com/LINCnil/GDPR-Developer-Guide), and the EDPB, ENISA and ESAs documents under their publishers' reuse terms.

## Parsing

Each document is split into blocks tagged with their place in the document, so that answers can cite it: *GDPR, Article 33(1)*, *EDPB Guidelines 9/2022, II. > A. > 1. Article 33 requirements, p. 10*.

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

- **Three retrievers**, picked by the pipeline file (see [Pipelines](#pipelines)):
  - **vector** (the baseline): the chunks whose embedding is nearest to the question's (cosine).
  - **bm25**: the chunks sharing the most words with the question, rare words weighing more, with
    [pg_textsearch](https://github.com/timescale/pg_textsearch) (English stemming and stop words).
    It finds the exact terms that embeddings blur, such as a defined term or an article's wording.
  - **hybrid**: both, fused by [reciprocal rank fusion](https://doi.org/10.1145/1571941.1572114)
    (RRF): a chunk scores the sum of 1 / (60 + rank) over the rankings that have it. Only ranks count,
    since BM25 and cosine scores cannot be compared, so a chunk both retrievers find comes first. Each
    retriever returns the same 50 candidates whatever the number of sources asked for, so the top 8
    the evaluation measures (from a top 50) is exactly the top 8 an answer gets. Measured, it does
    worse than vector search alone on the golden set (see [Results and findings](#results-and-findings)).

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

Known limitations, to be measured by the evaluation: no reranking yet, a fixed number of sources,
and no access control on the documents yet.

## Pipelines

A pipeline is the recipe of one variant of the RAG: which bricks answer a question. Each variant is a
TOML file in [`pipelines/`](pipelines/), so comparing two systems needs no code change:

```toml
# pipelines/hybrid.toml
retriever = "hybrid"
```

```bash
uv run agentic-rag ask --pipeline pipelines/hybrid.toml "Who must test the ICT business continuity plans?"
uv run agentic-rag eval --pipeline pipelines/hybrid.toml   # saved as eval/results/hybrid.json
```

Without `--pipeline`, the defaults are the baseline (vector search). A misspelled key is refused
rather than silently running the baseline under another name, and the validated pipeline, defaults
included, is saved with each evaluation run, so a result always says which system produced it. The
file holds only what a brick needs: fields appear with the bricks (reranker, guardrails, LLM), while
secrets and infrastructure stay in `.env`.

## Tracing

With the keys of a [Langfuse](https://langfuse.com) project in `.env` (`LANGFUSE_*`), each
question becomes a trace:

```text
ask                    the question; the answer and the sources it cites
├── retrieve           the k chunks found, with their citations and scores
│   └── embedding      the question's embedding: tokens, cost
└── generation         the full prompt, the model, tokens (reasoning included), cost
```

A trace tells a retrieval miss (the right article is not among the sources) from an LLM mistake,
and shows where the time goes: on the first traces, embedding the question took 1.3 to 2.6 s and the
vector search ~50 ms. `ask` and `retrieve` come from wrappers that `factory.py` adds around the RAG
and its retriever, so their code knows nothing of Langfuse; Langfuse's OpenAI integration traces
the OpenAI calls. Without the keys, nothing is traced or sent. With them, the questions, sources and
answers are sent to Langfuse: masking personal data is planned with the guardrails (Phase 5).

## Evaluation

Every later change (hybrid search, reranking, other chunking) is kept only if it improves a
measure, so the measure comes first. [`eval/golden.toml`](eval/golden.toml) is the golden set: 104
questions, each with the passages of the corpus that answer it.

| Category | Questions | What it tests |
|---|---|---|
| factual | 56 | a precise fact in one text |
| vague | 26 | a question as a user asks it, without the words of the article |
| table | 5 | facts that only a table of a PDF states |
| multi-document | 9 | an answer that needs two documents |
| unanswerable | 8 | a question the corpus does not cover: the right answer is the refusal |

- **Passages, not chunk ids.** The evidence of a question is a short verbatim quote of the
  corpus. A retrieved chunk is relevant when it contains the quote whole (case, punctuation and
  spacing ignored), so the golden set stays valid when the parsing or the chunking changes, which is
  exactly what the later phases do. A fact can list several quotes (`any_of`) when documents state it
  in different words: a guideline that quotes the GDPR answers as well as the GDPR.
- **Written two ways.** 40 questions were written while reading a passage, 64 blind, as a user
  would, without reading the corpus. Comparing their scores shows whether the first kind is too easy.
- **A held-out fifth.** 21 questions (a fifth of each category) are never looked at while tuning,
  to check at the end that the tuning did not just fit the other questions. Retrieval is tuned on the
  77 others that have facts to find (unanswerable questions are left to the generation metrics).
- **Checked by tests.** `tests/evaluation/test_golden.py` verifies the structure (unique ids,
  documents of the manifest, an unanswerable question has no evidence and expects the refusal) and
  that each quote is whole in a chunk of its document and not so generic that it matches many chunks
  (these two read the corpus: they are skipped until `download` and `convert` have run).

Retrieval metrics, per question:

- **recall@k** (the main one at k = 8, the number of sources the LLM gets): the share of the
  question's facts that the first k chunks state. Also at 5, 20 and 50, to see whether a missed fact
  is ranked too low or not found at all.
- **MRR@10**: 1 / the rank of the first chunk that states a fact, 0 if none of the first 10 does.

```bash
uv run agentic-rag eval --pipeline pipelines/hybrid.toml   # run, saved as eval/results/hybrid.json
uv run agentic-rag compare eval/results/baseline.json eval/results/hybrid.json
```

A run records the commit, whether the tree had uncommitted changes, the hash of the golden set and
the pipeline. `compare` refuses two runs on different golden sets, and two systems are compared
question by question: the mean of the paired differences, with a 95 % bootstrap confidence interval.
Pairing compares each question with itself, so the gap in difficulty between questions does not
blur the comparison. With 77 questions, both a score and a difference are known to about ±8 points:
smaller effects need more questions.

Still to do: pooling (judging the chunks that several retrievers rank high, to find relevant ones
the quotes do not list), the generation metrics (faithfulness, answer relevancy, correct refusals)
with an LLM judge calibrated on human grades, and thresholds in CI.

## Results and findings

What each measured change did, how we read it and what we decided, in the order the project went.
The numbers come from the runs saved in [`eval/results/`](eval/results/) and can be recomputed with
`compare`. Unless stated otherwise: retrieval on the 77 tuning questions, differences against the
previous system, **bold** when the 95 % interval excludes 0.

**Current system:** vector search (entry 1). Hybrid search is available as a pipeline but not the
default (entry 2).

### 1. Baseline: vector search

| recall@5 | recall@8 | recall@20 | recall@50 | MRR@10 |
|---|---|---|---|---|
| 0.59 | 0.72 [0.63, 0.80] | 0.82 | 0.90 | 0.57 [0.48, 0.66] |

| recall@8 | factual (45) | vague (21) | multi-document (7) | table (4) | origin: passage (31) | origin: blind (46) |
|---|---|---|---|---|---|---|
| vector | 0.79 | 0.59 | 0.44 | 1.00 | 0.88 | 0.61 |

Origin: written while reading the passage that answers it, or blind, before looking at the corpus.

- **The missing facts are mostly ranked too low, not lost.** 28 % of the facts are not in the top 8,
  but recall@50 is 0.90: for many questions the right chunk is retrieved, between rank 9 and 50. A
  reranker, which reorders a short list with a finer model, targets exactly this gap.
- **Blind questions are much harder** (0.61 against 0.88). Questions written while reading a passage
  reuse its words, which makes them easy; the blind ones are closer to what users ask, so both are
  reported.
- **Multi-document questions** (0.44) need two texts in the same 8 chunks. With 7 questions (and
  4 tables), these categories are indications, not results.

### 2. BM25 and hybrid search: not adopted

| | vector | bm25 | hybrid (RRF) | hybrid − vector |
|---|---|---|---|---|
| recall@8 | 0.72 | 0.44 | 0.56 | **−0.16 [−0.24, −0.07]** |
| recall@50 | 0.90 | 0.65 | 0.87 | −0.03 [−0.08, +0.02] |
| MRR@10 | 0.57 | 0.34 | 0.47 | **−0.10 [−0.18, −0.02]** |
| recall@8, factual (45) | 0.79 | 0.56 | 0.68 | |
| recall@8, vague (21) | 0.59 | 0.20 | 0.26 | |

- **BM25 is weak on this golden set, by design.** It only finds the words of the question, and the
  rule of the golden set is that a question names its text but not the wording of the article. On
  vague questions, worded as a user would, it finds 0.20 of the facts.
- **Fusing it in hurts more than it helps**: 20 questions lose recall@8, 7 gain. With RRF's usual
  constant (60) and 50 candidates per retriever, a chunk ranked middling by both (14th and 11th:
  1/74 + 1/71 ≈ 0.028) beats a chunk only vector search ranks first (1/61 ≈ 0.016). For instance the
  AI Act recital on disclosing a chatbot is 1st in vector search and 12th in the hybrid ranking.
- **No setting fixes it.** Re-fusing the saved rankings offline (checked to reproduce the hybrid
  run exactly), we tried 150 settings: the RRF constant (1 to 60), the candidates per retriever
  (8 to 50) and a weight on BM25 (0 to 1). With equal weights, the best still loses
  (−0.073 [−0.145, −0.002]). The best overall gives BM25 a quarter of the weight, with a constant of
  1 and 10 candidates: +0.022 [0.000, +0.050], 3 questions gained and none lost, MRR unchanged. That
  is the best of 150 tries on the same questions, so it overstates the gain; checking it on the 21
  held-out questions could not detect 2 points.

**Decision:** vector search stays the default, and hybrid search stays as a pipeline to compare
later bricks with. BM25 is kept to find candidates for pooling.

**What this does not show:** that BM25 is useless. It fails on paraphrased questions, while users
of a compliance assistant also type exact terms ("Article 23", "TLPT", "ICT third-party service
provider"), where word matching is known to help. Next: pooling, since relevant chunks the hybrid
ranking brings up but the quotes do not list currently count as misses; then a category of
questions in exact legal terms. The rule is set before running it: hybrid search becomes the
default if it gains significantly on that category without losing significantly on the others.

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
