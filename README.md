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
├── document.py     domain types: Division, Block, Chunk, StoredChunk, RetrievedChunk
├── ingestion/      corpus download, parsers/ (EUR-Lex, PDF, Markdown), chunking, pipeline
├── embeddings/     base.py (Embedder), openai.py
├── storage/        base.py (ChunkStore, VectorSearch, KeywordSearch), postgres.py
├── retrieval/      base.py (Retriever), vector.py, bm25.py, hybrid.py
├── generation/     base.py (Generator, Answer), prompt.py, openai.py
├── citation.py     how a chunk is referred to: "GDPR, Article 33(1)"
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

What gets embedded is set by the pipeline (`embedded_text`): the chunk's text alone (the baseline),
or after its document title and titled path (the default since entry 5 of the results), or after a
context an LLM writes. Only the vectors change: BM25, the LLM and the metrics read the chunk's text.
Each choice is ingested into its own PostgreSQL schema (`corpus`), so the variants live side by side.

Known limitations:

- A single line over the budget stays whole: a recital is one line, up to 833 tokens, well below
  the 8191 tokens the embedding model accepts.
- In PDFs, nesting only shows in the bullets, so a split can separate "...the following:" from
  its list (3 cases in the corpus).
- The pieces of a split PDF block all carry the page the block starts on.

## Retrieval and answers

- **Three retrievers**, picked by the pipeline file (see [Pipelines](#pipelines)):
  - **vector** (the default): the chunks whose embedding is nearest to the question's (cosine).
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

- **HyDE** (optional, `hyde = "ollama"`): an LLM first writes a short answer to the question
  without any document, worded like a regulation, and the search uses that answer instead of the
  question ([Gao et al., 2022](https://arxiv.org/abs/2212.10496)). A reranker, when there is one,
  still scores the chunks against the real question. The answers are cached in
  [`eval/hyde.jsonl`](eval/hyde.jsonl), so the evaluation runs can be redone the same. Measured, not
  adopted (see [Results and findings](#results-and-findings)).

Known limitations, to be measured by the evaluation: a fixed number of sources,
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

Without `--pipeline`, the defaults are the best composition adopted so far (vector search on the
titled chunks), saved as `eval/results/default.json`; [`pipelines/baseline.toml`](pipelines/baseline.toml)
keeps the plain RAG the ablation starts from. A variant file only states what differs from the
defaults, so a new brick stacks on what was adopted before it. A misspelled key is refused rather than
silently running the defaults under another name, and the validated pipeline, defaults included, is saved with each evaluation run, so a result always says which system produced it. The
file holds only what a brick needs: fields appear with the bricks (reranker, guardrails, LLM), while
secrets and infrastructure stay in `.env`. Each LLM role names who plays it (`contextualizer`,
`hyde`, `generator`: `"openai"` or `"ollama"`); which model that is stays in `.env`.

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
- **MRR@8**: 1 / the rank of the first chunk that states a fact, 0 if none of the first 8 does
  (MRR@10 until the judged@8 runs: ranks 9 and 10 are neither given to the LLM nor judged).
- **judged@8**: the share of the first 8 chunks that have a judgment: they state a fact, or
  `eval/judgments.toml` rejects them for this question. The recall counts an unjudged chunk as
  useless, so a low judged@8 warns that the recall may be underestimated. Divided by the chunks
  retrieved when fewer than 8, as ir-measures' Judged does.

```bash
uv run agentic-rag eval --pipeline pipelines/hybrid.toml   # run, saved as eval/results/hybrid.json
uv run agentic-rag compare eval/results/baseline.json eval/results/hybrid.json
```

A run records the commit, whether the tree had uncommitted changes, the hashes of the golden set and
of the judgments, and the pipeline. `compare` refuses two runs on different golden sets (and warns
on different judgments), and two systems are compared question by question: the mean of the paired
differences, with a 95 % bootstrap confidence interval.
Pairing compares each question with itself, so the gap in difficulty between questions does not
blur the comparison. With 77 questions, both a score and a difference are known to about ±8 points:
smaller effects need more questions.

**Pooling** finds the relevant chunks the quotes do not list. `pool` takes the first 8 chunks of
several runs that nothing judges yet and has an LLM pre-judge them, each verdict with the facts it
found and their words. A person then checks, blind, all the chunks it found relevant and 100 of its
rejections drawn at random. `merge-pool` pours the result into the golden set and the judgments.

```bash
uv run agentic-rag pool eval/results/baseline.json eval/results/bm25.json eval/results/hybrid.json
# fill in eval/pooling/review.toml, then:
uv run agentic-rag merge-pool
```

Still to do: the generation metrics (faithfulness, answer relevancy, correct refusals) with an LLM
judge calibrated on human grades, and thresholds in CI.

## Results and findings

What each measured change did, how we read it and what we decided, in the order the project went.
The numbers come from the runs saved in [`eval/results/`](eval/results/) and can be recomputed with
`compare`. Unless stated otherwise: retrieval on the 77 tuning questions, differences against the
previous system, **bold** when the 95 % interval excludes 0.

**Current system:** vector search on chunks embedded after their titles (entry 5). Hybrid search,
reranking, contextual retrieval and HyDE are available as pipelines but not the default (entries 2,
4, 5, 7).
Entries keep the numbers of their time: entry 3 corrects those of 1 and 2. Entries 1 to 4 were
measured on the plain baseline; since entry 5, the variants that only set the retriever stack on the
titles. Entry 6 redoes BM25 and hybrid search on them; the reranking run is still from entry 4.

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
  (−0.073 [−0.137, −0.005]). The best overall gives BM25 a quarter of the weight, with a constant of
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
(Entry 3 explains why this rule was dropped.)

### 3. Pooling: most of the unread chunks were not relevant

Before this entry, only 16 % of the baseline's top 8 had been judged (judged@8), so its recall could
hide relevant chunks the quotes did not list. We pooled the top 8 of the vector, BM25 and hybrid runs:
1092 (question, chunk) pairs that nothing judged. `gpt-6-luna` pre-judged them all (55 relevant, 1037
rejected); one person checked, blind, those 55 and 100 random rejections.

| The LLM against the person | count | 95 % interval (Wilson) |
|---|---|---|
| rejections the person found relevant | 0 / 100 | 0 to 3.7 % |
| "relevant" verdicts the person confirmed | 37 / 55 (67 %) | 54 to 78 % |
| facts missing from the golden set | 0 | |

**The rule was set before opening the review:** keep the LLM's unchecked rejections if the upper
bound of its false rejections is at most 10 %. It passed. So the golden set gained 54 wordings from
the 37 confirmed chunks, and the judgments 1054 rejections (118 checked by hand, 936 kept by the rule).

| | recall@8 before | after | gain | MRR@8 before | after |
|---|---|---|---|---|---|
| vector | 0.72 | 0.75 [0.68, 0.83] | **+0.04 [+0.01, +0.08]** | 0.57 | 0.63 |
| bm25 | 0.44 | 0.48 | **+0.04 [+0.01, +0.07]** | 0.33 | 0.37 |
| hybrid | 0.56 | 0.61 | **+0.05 [+0.02, +0.10]** | 0.46 | 0.53 |

| recall@8, after | factual (45) | vague (21) | multi-document (7) | table (4) |
|---|---|---|---|---|
| vector | 0.82 | 0.67 | 0.44 | 1.00 |
| hybrid | 0.73 | 0.34 | 0.44 | 1.00 |

- **The recall was underestimated, but not by much.** Only 3.4 % of the pooled pairs are relevant,
  on 20 of the 77 questions. The low judged@8 mostly meant "not read yet".
- **It does not rescue hybrid search**, which entry 2 left as an open question. It gains slightly more
  than vector search, but still loses **−0.14 [−0.22, −0.06]** on recall@8 and
  **−0.10 [−0.19, −0.02]** on MRR@8, with 20 questions lost and 6 gained.
- **The LLM over-detects, which is the right error here.** It missed nothing in the sample, and its
  false "relevant" verdicts (often the right topic in the wrong regulation) cost only reading time,
  since all of them are checked. It would not do as a judge on its own.
- **A chunk can state several facts** (one DORA recital states all five of its question). The review
  first allowed one per chunk, which would have undercounted the recall, so it now takes a list.

**Decision:** the LLM's rejections are kept, and vector search stays the default.

**What this does not show:**
- judged@8 is now 1.00, but only because the pool is these three runs' top 8. A new brick will bring
  up unread chunks and lower it, so each one needs its own pooling round.
- Ranks 9 to 50 were not pooled, so recall@20 and recall@50 stay underestimated.
- There is one reviewer, so no measure of agreement between people. The agreement on which facts a
  chunk states (29 of 37) is not blind: facts were added to 9 chunks after the review.
- At worst, about 35 of the 936 unchecked rejections could be relevant chunks.
- The 21 held-out questions are not pooled, to keep them untouched.

**A rule corrected before use.** The rule of entry 2 can no longer pass, whatever the exact-term
questions show: after pooling, hybrid search already loses significantly on factual questions
(−0.085 [−0.178, −0.015] on recall@8) and vague ones (−0.333 [−0.520, −0.147]). The only open
question is whether to send just the exact-term queries to word matching, a routing decision. So
that category of questions moves to just before the routing work.

**What this leaves open:** the golden set still has no question in exact legal terms, so later
bricks are measured without them and will have to be rerun once it does. Article numbers also need
care: they are in a chunk's path, not its text, so neither retriever sees them.

Next: the reranker.

### 4. Reranking: not adopted, too slow on this machine

A reranker is a finer model that reads the question and each candidate together, then reorders the
candidates. Here it reorders the top 50 of vector search, a depth chosen because recall@8 is 0.75
and recall@50 is 0.90: at most 0.15 to win.

| | vector | + bge-reranker-v2-m3 | difference |
|---|---|---|---|
| recall@8 | 0.75 | 0.73 | −0.02 [−0.09, +0.05] |
| recall@5 | 0.64 | 0.67 | +0.03 [−0.06, +0.12] |
| recall@20 | 0.84 | 0.85 | +0.01 [−0.05, +0.08] |
| MRR@8 | 0.63 | 0.66 | +0.03 [−0.06, +0.12] |
| judged@8 | 1.00 | 0.69 | **−0.31 [−0.36, −0.27]** |
| recall@8, factual (45) | 0.82 | 0.84 | |
| recall@8, vague (21) | 0.67 | 0.52 | |
| recall@8, multi-document (7) | 0.44 | 0.56 | |
| recall@8, table (4) | 1.00 | 1.00 | |

- **No measurable gain on recall@8**: 10 questions gain, 11 lose. Recall@50 is identical, as it
  should be: the reranker only reorders the same candidates.
- **These figures lean against the reranker.** It brings up 193 chunks nobody has judged (out of
  616), and they count as misses. The difference on recall@8 is therefore a lower bound.
- **The model matters.** In one-off runs outside the repo, with the same unjudged-chunk bias, two
  light models lost clearly: `ms-marco-MiniLM-L6-v2` −0.15 [−0.25, −0.06] and
  `jina-reranker-v1-turbo-en` −0.20 [−0.30, −0.10]. `v2-m3` beat both by +0.13 and +0.18, both
  significant.
- **Speed decides it.** Reranking 50 candidates on CPU, in those one-off runs:

| model | p50 | p95 | memory peak |
|---|---|---|---|
| `bge-reranker-v2-m3` | 77 s | 188 s | 2.84 GB |
| `ms-marco-MiniLM-L6-v2` | 2.9 s | 3.5 s | 0.87 GB |
| `jina-reranker-v1-turbo-en` | 5.1 s | 6.7 s | 1.48 GB |

**Decision:** no reranker by default, and `pipelines/reranker.toml` stays to compare later bricks
with. Adopting one required a 95% interval on recall@8 entirely above zero, set before any run, and
a p95 under 10 s per question on this machine, set once a first trial had shown about 70 s per
question for `v2-m3`. No model passes both: the only one that keeps the recall is nearly 19 times too slow.

**No pooling this time**, unlike the plan. Judging those 193 chunks could not change the decision:
`v2-m3` fails on latency whatever its recall. At the last round's rate (54 relevant out of 1108)
they would hold about ten relevant chunks, while the interval needs at least +0.09 to clear zero.

**What this does not show:** that `v2-m3` is no better than vector search, since the unjudged chunks
leave that open, or anything about reranking on a GPU, where the latency verdict may not hold and
the pooling round would be worth doing. The golden set also still has no question in exact legal
terms, so this run will have to be redone once it does.

### 5. Titles and contextual retrieval: titles adopted, contextual retrieval not

Vector search only sees a chunk's own text, but a chunk often makes sense only in its place: "1. In
the case of a personal data breach, the controller shall…" does not say it is the GDPR, Article 33.
Two ways to put that back into what is embedded (the stored text does not change):

- **titles**, no LLM: the document title and the titled path before the text.
  `Regulation (EU) 2016/679 (General Data Protection Regulation) > Chapter IV: Controller and
  processor > Section 2: Security of personal data > Article 33: Notification of a personal data
  breach to the supervisory authority`
- **contextual retrieval** ([Anthropic, 2024](https://www.anthropic.com/news/contextual-retrieval)):
  an LLM reads the whole document and writes one or two sentences that situate each chunk. *"The
  GDPR, Article 33, which is part of Chapter IV (Controller and processor) and Section 2 (Security of
  personal data), outlines the controller's obligation to notify the supervisory authority of a
  personal data breach within 72 hours…"* The 1922 contexts were written by `qwen3.8:27b` through
  Ollama at temperature 0 (median 44 words), and are cached in
  [`eval/contexts.jsonl`](eval/contexts.jsonl), so the ingestion can be redone without the model.

Titles is the witness: it tells what the LLM adds beyond the structure the parser already knows.
**The rule was set before the runs:** contextual retrieval is adopted if its 95 % interval on
recall@8 is entirely above 0 against both the baseline and titles; otherwise titles is kept.

The new chunks the two variants brought up were pooled first (a second round, same method as
entry 3): 302 unjudged pairs, 17 found relevant by the LLM, 117 checked blind by a person. 1 of 100
sampled rejections was wrong (upper bound 5.4 %), so the unchecked rejections were kept; one review
found a fact the golden set missed, added by hand. All three runs are then at judged@8 1.00, and the
differences barely moved (±0.01), so the gains are not an artifact of unjudged chunks.

| | baseline | titles | contextual | titles − baseline | contextual − baseline | contextual − titles |
|---|---|---|---|---|---|---|
| recall@5 | 0.64 | 0.73 | 0.72 | **+0.09 [+0.03, +0.16]** | **+0.08 [+0.01, +0.16]** | −0.01 [−0.06, +0.05] |
| recall@8 | 0.75 | 0.77 | 0.81 | +0.02 [−0.02, +0.06] | +0.05 [−0.01, +0.12] | +0.04 [−0.02, +0.10] |
| recall@50 | 0.90 | 0.93 | 0.96 | +0.03 [−0.00, +0.07] | **+0.06 [+0.02, +0.11]** | +0.03 [−0.01, +0.07] |
| MRR@8 | 0.63 | 0.68 | 0.71 | +0.05 [−0.02, +0.12] | **+0.08 [+0.00, +0.16]** | +0.03 [−0.04, +0.10] |
| recall@8, factual (45) | 0.82 | 0.84 | 0.86 | | | |
| recall@8, vague (21) | 0.67 | 0.67 | 0.81 | | | |
| recall@8, multi-document (7) | 0.44 | 0.44 | 0.37 | | | |
| recall@8, table (4) | 1.00 | 1.00 | 1.00 | | | |

- **Titles move the right chunk up, for free.** Recall@5 gains 0.09 (15 questions gain, 5 lose),
  but by rank 8 the baseline has mostly caught up (6 gain, 4 lose).
- **Contextual retrieval finds more, but not significantly at rank 8.** Recall@8 gains 0.05 over the
  baseline (14 questions gain, 5 lose) and 0.04 over titles (9 gain, 4 lose), both intervals
  touching 0. It is significant deeper (recall@50 +0.06: chunks the baseline does not find at all)
  and on the ranking (MRR@8 +0.08).
- **The gain is on vague questions** (0.67 to 0.81), worded as a user would: the context restates
  in plain words what a provision is about. Titles do nothing there (0.67). With 21 questions this
  is an indication, not a result.

**Decision:** titles become the default (`pipelines/baseline.toml` keeps the plain RAG). Contextual
retrieval is not adopted: the rule is not met, and it costs an LLM call per chunk at every
ingestion, from a model that is not available everywhere. It stays as `pipelines/contextual.toml`.

**What this does not show:**
- That contextual retrieval does not help: +0.04 over titles is below what 77 questions can detect
  (about ±8 points). A larger golden set would settle it, and it will be rerun then.
- Anything about article numbers: titles put "Article 33" into the embedded text, but the golden set
  has no question in exact legal terms yet (entry 3).
- Whether contexts that summarize the passage, rather than only situate it, help or hurt: some do,
  despite the prompt, and this was not measured separately.
- Anything about other contextualizing models: only Qwen wrote these contexts.

### 6. BM25 and hybrid search on the titled chunks: still not adopted

Entry 5 changed what the vector search embeds, so the BM25 and hybrid runs of entries 2 and 3 no
longer compared with the default. Both were rerun with it at the same commit, on the golden set and
judgments of entry 5.

| | default | bm25 | hybrid | bm25 − default | hybrid − default |
|---|---|---|---|---|---|
| recall@5 | 0.73 | 0.44 | 0.57 | **−0.29 [−0.39, −0.20]** | **−0.16 [−0.25, −0.08]** |
| recall@8 | 0.77 | 0.47 | 0.67 | **−0.30 [−0.39, −0.20]** | **−0.10 [−0.17, −0.04]** |
| recall@20 | 0.86 | 0.58 | 0.83 | **−0.28 [−0.38, −0.19]** | −0.03 [−0.09, +0.02] |
| recall@50 | 0.93 | 0.68 | 0.91 | **−0.25 [−0.35, −0.16]** | −0.02 [−0.07, +0.01] |
| MRR@8 | 0.68 | 0.37 | 0.55 | **−0.31 [−0.42, −0.20]** | **−0.12 [−0.22, −0.04]** |
| judged@8 | 1.00 | 1.00 | 0.91 | | |
| recall@8, factual (45) | 0.84 | 0.61 | 0.78 | | |
| recall@8, vague (21) | 0.67 | 0.20 | 0.41 | | |
| recall@8, multi-document (7) | 0.44 | 0.32 | 0.51 | | |
| recall@8, table (4) | 1.00 | 0.62 | 1.00 | | |

- **BM25 did not change**, as expected: it reads the stored text, which has no titles. Its top 8 is
  the same as before for all 77 questions; recall@8 moves from 0.44 to 0.47 only through the
  judgments added by pooling.
- **Hybrid search still loses at the top** (recall@8: 3 questions gain, 15 lose), and again on vague
  questions (0.67 to 0.41). Deeper, the loss is no longer significant (recall@50 −0.02): the fusion
  keeps the right chunks among the candidates but ranks BM25's wrong ones above them.
- 9 % of the hybrid top 8 was never judged and counts as missed. It was not pooled again: in the
  last round, 3 to 6 % of such chunks were relevant, about +0.01 of recall, far from the −0.10.

**Decision:** unchanged, vector search stays the default.

**What this does not show:** anything on questions in exact legal terms, which the golden set still
lacks (entry 2). The multi-document gain (0.44 to 0.51) rests on 7 questions.

### 7. HyDE: not adopted

The weakest questions are the vague ones (recall@8 0.67 against 0.84 for factual ones): worded as a
user would, they share few words with the texts, and the right chunk is often found but ranked too
low (recall@50 0.93 against 0.77 at 8). [HyDE](https://arxiv.org/abs/2212.10496) answers first and
searches with the answer: an LLM (`qwen3.8:27b` through Ollama, temperature 0) writes two to four
sentences "as the text of an EU regulation or of official guidance would", without article numbers,
and only that answer is embedded. *"Our privacy person keeps blocking projects and management wants
to let her go for it. Are we allowed to?"* becomes *"The data protection officer shall not be
dismissed or penalised for performing his or her tasks…"*, almost the wording of GDPR Article 38(3),
which the default search ranks below 20.

**The rule was set before the run:** HyDE is adopted if its 95 % interval on recall@8 is entirely
above 0 against the default, after pooling. The first run left 42 % of its top 8 unjudged, so its
new chunks were pooled (third round, same method as entry 3): 257 pairs, 28 found relevant by the
LLM, 128 checked blind by a person; 2 of 100 sampled rejections were wrong (upper bound 7.0 %), so
the unchecked rejections were kept. The review also found a fact the golden set missed, added by hand,
and a table question whose facts a prose chunk states too, now counted as factual. Both runs were
then redone at the same commit, at judged@8 1.00.

| | default | HyDE | HyDE − default |
|---|---|---|---|
| recall@5 | 0.73 | 0.77 | +0.04 [−0.07, +0.15] |
| recall@8 | 0.77 | 0.83 | +0.06 [−0.03, +0.16] |
| recall@20 | 0.86 | 0.87 | +0.01 [−0.07, +0.09] |
| recall@50 | 0.93 | 0.91 | −0.02 [−0.09, +0.05] |
| MRR@8 | 0.68 | 0.75 | +0.08 [−0.03, +0.18] |
| recall@8, factual (46) | 0.84 | 0.86 | |
| recall@8, vague (21) | 0.67 | 0.85 | |
| recall@8, multi-document (7) | 0.44 | 0.56 | |
| recall@8, table (3) | 1.00 | 0.75 | |

- **It helps where it was aimed.** Vague questions rise from 0.67 to 0.85, the level of factual
  ones; the DPO dismissal, DPO conflict of interest and TLPT frequency questions go from 0 to 1. Even
  a wrong answer can help: Qwen wrote that a penetration test is due "at least once a year" (DORA
  says every 3 years for TLPT), and its words still found the right article.
- **It loses on concrete cases.** 16 questions gain and 11 lose. The answer reads like a
  regulation, so the search finds articles and recitals where the default found the EDPB or CNIL
  guidance that applies them: the 18 relevant chunks the pooling added were all outside the default
  top 8, 12 of them GDPR provisions. The credential breach question falls from 1.00 to 0.25, three
  others from 1 to 0.
- **Not significant.** The gain is positive in 89 % of the bootstrap samples, but the interval
  touches 0, as with contextual retrieval (entry 5): 77 questions cannot tell an effect of 5 or
  6 points (about ±8 points).

**Decision:** not adopted. The rule is not met, and HyDE costs an LLM call before every search (0.4
to 8 s measured), outside the cache once real users ask new questions. It stays as
`pipelines/hyde.toml`.

**What this does not show:**
- How much rests on the review's reading that a general rule states a fact (GDPR Article 34 for
  "inform the patients" in the hospital case), applied as in earlier rounds; a stricter reading would
  lower HyDE on those questions.
- Anything about unanswerable questions, where the LLM invents a plausible answer and may bring up
  misleading chunks: that needs the evaluation of the answers.
- Another prompt: one that also asks for the wording of guidance, or searching with both the
  question and the answer, fused, might keep the gains without the losses. Next: that fusion, measured
  on a larger golden set.
- Only one draw of Qwen per question: the answers are cached, so the runs repeat, but another draw
  could rank differently.

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
