<h1 align="center">The Library</h1>

<p align="center">
  A personal research library that has actually read its books.<br/>
  Drop in papers, talk to the collection, and see in the margin exactly which page each claim came from.
</p>

<p align="center">
  <img src="docs/web.jpg" alt="The Ask pane. Behind it, the library drawn as a nebula: one point per volume, edges where volumes cite each other, share a thread, or sit close in meaning. On the left, the rack with an inserted cartridge and the two-level shelf." width="900"/>
</p>

<p align="center">
  <img src="docs/ask.jpg" alt="An answer. A reading column with markdown rendered; every citation resolves into a note in the margin beside the block that cites it, and the volumes it drew on light up in the nebula behind." width="900"/>
</p>

<p align="center">
  <em>Runs entirely on your own hardware. No cloud, no API keys, no telemetry.</em>
</p>

---

## What it is

Most "chat with your documents" tools are a search box with a language model bolted on: embed the chunks, retrieve the nearest ones, hope the answer is in there. The Library does that too — but it also **reads**. Every document goes through a background pass that writes a summary of each section, extracts the claims it makes, and tags its subjects. Documents you care about get a deeper pass that writes **marginalia**: what a thoughtful reader thinks while reading each passage, not a restatement of it.

Then the collection is treated as one thing. Claims are clustered *across* documents so a theme running through five papers becomes a single, readable entry. Citations between papers are extracted and matched. Where two sources genuinely disagree, the Library says so.

And when you ask it something, the answer is a reading column with an apparatus: each `[n]` resolves into a note in the margin naming the volume, page, and section it came from. Those markers are **verified after generation** — a citation the model invents is stripped rather than shown — so a flash of red always means provenance. Anything unmarked is the librarian's own reasoning, and it's meant to reason: the answering policy is deliberately open.

<p align="center">
  <img src="docs/reader.jpg" alt="Reading a volume. The section summary is the italic lead; the library's reflection sits in the margin beside the passage it is about." width="900"/>
</p>

## Using it

You need Postgres 16, Redis, [Ollama](https://ollama.com), and [uv](https://docs.astral.sh/uv/) on the machine, and about 20 GB of RAM wherever Ollama runs — a second box over an SSH tunnel is fine.

```sh
./scripts/bootstrap.sh          # once: pgvector, extensions, models, migrations
./library                        # checks services, migrates, starts worker + API, opens the browser
```

`./library stop`, `restart`, and `status` do what they say. `uv run python scripts/verify.py` exercises every surface against the running instance — ingest, read, annotate, search, chat on both models, the library layer, delete — and reports pass/fail per check. Ctrl-C in the foreground closes everything.

Drop PDFs, Markdown, or text onto the shelf — folders are walked. A document is **searchable within seconds**. Click *have it read* for section summaries and subjects (a couple of minutes), then *annotate it* for marginalia. For a first load:

```sh
./library import ~/papers ~/books --read     # walks directories, skips what's already shelved
curl -X POST 'localhost:8077/api/read/backfill?tier=2'   # annotate everything once read
```

**Ask** — talk to the collection. Narrow it by subject with the chips or by clicking a subject on the shelf. Switch models per conversation. Set a **stance** to loosen the librarian's reserve: *opinionated*, *contrarian · charitable*, *cynical · optimistic*. Answers given under a stance are labelled in violet so you always know which ones were the librarian speaking for itself.

**Find** — plain retrieval, showing each passage's dense and lexical rank.

**Threads** — themes spanning volumes, where your sources disagree, and who cites whom. Rebuilt automatically a few minutes after the last read finishes, so a folder of forty papers produces one rebuild rather than forty.

<p align="center">
  <img src="docs/threads.jpg" alt="Threads across the collection: themes drawn from claims in several volumes, and the citation graph." width="900"/>
</p>

## How it works

**Reading is a quality level, not a pipeline stage.**

| Tier | What happens | Cost | Result |
|---|---|---|---|
| 0 · *listed* | extract → section tree → chunk → embed | seconds, no LLM | searchable |
| 1 · *read* | orientation card, one structured call per section, document summary, subjects | ~2 min / paper | summaries as section leads |
| 2 · *annotated* | one structured call per section writing a reflection for each passage | ~1.5 min / paper | marginalia in the reader |

This is what makes a 500-document backfill a single overnight rather than the ~260 GPU-hours a per-chunk deep pass would cost.

**Retrieval** is dense + lexical fused with reciprocal rank fusion, in one Postgres query, then reranked by a cross-encoder — every stage measured against a generated question set and switchable. Chat reranks; keyword search doesn't, because the eval showed the cross-encoder *hurts* short keyword queries.

**The shelf is two levels, and every volume sits in one place.** Tier 1 tags each document with up to four subjects, which is right for finding things and wrong for shelving them. So shelving is a separate pass: the model organises the collection into a handful of *top shelves* (fields — Cybersecurity, Machine Learning) each with *sub-shelves* named from the titles actually on them, then files every volume on exactly one sub-shelf. Crowded sub-shelves are split from their own titles; sub-shelves with a volume or two are dissolved into their neighbours. New volumes are shelved as they are read. *Reshelve* redoes the whole thing; the tags remain for filtering.

**The library layer** clusters extracted claims (not section summaries — measured: summaries cluster with their own paper, claims cluster across papers) and summarises each cluster once. Citations are parsed from reference sections and matched by title. Contradiction detection runs over cross-document clusters only.

**One store.** Postgres with `pgvector` holds documents, sections, chunks, artifacts, vectors, and the lexical index. Vectors commit in the same transaction as the rows they describe, so there is nothing to reconcile. Every generated artifact records the model and prompt version that produced it, so changing a prompt regenerates only what that prompt owns.

**Extraction does the unglamorous work.** Two-column papers are read column by column. Figure labels, axis ticks and legend text are dropped by position and font size rather than by regex. Footnotes are lifted out of the flow and placed after the page's prose with their numbers; the superscript markers they leave in the body are removed. Running headers are detected by frequency and stripped. Papers with no PDF bookmarks get their section tree from numbered headings in the text — which, it turns out, is most of them.

**Behind the Ask pane is the library itself**, drawn as a nebula: one point per volume, edges where volumes cite each other, share a thread, or sit close in meaning, laid out by a small force simulation in three dimensions and turning slowly. It thickens as the library grows, and the volumes an answer drew on light up as they are retrieved. Answers and markdown volumes are rendered as markdown; each block of an answer keeps its own margin notes.

## Settings and incidents

The Settings tab shows what the library is running on — services, models, retrieval and library configuration, each with the environment variable that changes it — and a maintenance row (collect garbage, rebuild threads, reshelve). Below it is the **incident log**: anything that escapes a route, fails a background job or a chat turn, or is logged at `ERROR` is recorded there, deduplicated within a window.

*Troubleshoot* asks the model to read an incident against the library's own documentation — [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md), this README, the devlog, the launcher and ops scripts — and answer with a diagnosis, the likely cause (environment, configuration, data, model, or a defect in the library itself), steps with commands for *you* to run, and the sections it leaned on. Nothing is executed: it says what to do; you do it. Every suggested command is checked against the commands the docs actually show, and anything the model appears to have invented is marked. When the cause looks like a defect, it drafts a GitHub issue — what happened, the error, what was tried, the environment — and offers it as a pre-filled link, with home paths and anything key-shaped scrubbed.

## Cartridges

A cartridge is a slice of a library that another library can put on its **rack**: a zip with a manifest, a colour, an icon, and the data. The receiving library doesn't just file it — it reads across everything it holds, so threads and disagreements form *between* collections, and every margin note keeps the colour of where it came from.

The point is sharing between people who can't share the documents. A confidentiality level sets what leaves:

| Level | Originals | Passages | Readings (summaries, notes) | What the receiver can do |
|---|---|---|---|---|
| `full` | yes | yes | yes | open the pages, cite them |
| `readings` | no | no | yes | search and cite *your reading* of each section; never sees the text |
| `catalogue` | no | no | summaries and subjects only | knows the material exists, asks you for it |

At `readings`, each section's summary stands in as its passage, so retrieval, the citation apparatus and the reader work unchanged — a note from such a source reads *Security's reading of …* rather than quoting a page. Levels control what leaves, not what happens after import.

```sh
./library export --name "Security" --level readings --subject "Network Security"
./library insert security-v1.zip           # or drop the zip on the accession box
```

<p align="center">
  <img src="docs/cartridge.jpg" alt="Making a cartridge: name, colour, icon, level, and a live preview of what would leave. On the left, the shelf open two levels deep: Cybersecurity, Deserialization Flaws, the volumes." width="900"/>
</p>

Click a spine on the rack to walk into that room: the composer becomes *Ask Security's shelf*, and Find, the shelf and follow-ups stay inside it. Eject removes what the cartridge brought and leaves what was already yours. A document that arrives from two cartridges is one document with two memberships; subjects merge by name; vectors ship as float16 and are loaded directly when the embedding model matches, re-embedded from the shipped text when it doesn't. Clusters and contradictions are never shipped — the receiver recomputes them across the new whole, which is the point. Content is hash-verified; there is no signing.

## Models

| Role | Default | Notes |
|---|---|---|
| Reading (all passes) | `qwen3:30b-a3b` | Pinned corpus-wide: artifacts from different models drift |
| Chat — general | `qwen3:30b-a3b` | Thinks before answering; the UI streams it |
| Chat — technical | `huihui_ai/qwen3-coder-abliterated` | Per-conversation toggle |
| Embeddings | `bge-m3` | 1024-dim, stored as `halfvec` |
| Reranker | `BAAI/bge-reranker-v2-m3` | In-process torch; the one thing not served by Ollama |

Ollama can be local or remote — every model call follows `LIBRARY_OLLAMA_URL`, which defaults to `localhost:11434`, so an SSH tunnel to a GPU box needs no configuration at all.

## What was measured

The interesting engineering came from measuring rather than assuming. The full log is in [`docs/DEVLOG.md`](docs/DEVLOG.md); the results that changed the design:

- **`websearch_to_tsquery` ANDs every term**, so a natural-language question matched *zero* rows and the lexical half of hybrid search was silently doing nothing. Fixed with an OR-semantics helper. Postgres full-text ranking also has no IDF term, which is why the lexical half is down-weighted rather than fused equally.
- **The cross-encoder helps paraphrased questions (recall@1 0.51 → 0.60) and hurts keyword queries (0.42 → 0.37).** Reranking depth 20 matches depth 100 on quality at a fifth of the latency.
- **A document-constant context prefix on every chunk made retrieval slightly worse**, not better — it homogenises that document's vectors. Reverted.
- **Tier 2 reflections don't improve retrieval**, on either a chunk-derived or a purpose-built interpretive question suite. They're kept because they're worth *reading*, and cut from the retrieval path.
- **Constrained JSON generation emits fields in schema order**, so a verdict placed before its reasoning is committed to before the model has reasoned. The contradiction detector returned `false` while its own explanation said "they report opposite effects". Reasoning fields come first, everywhere.
- **A too-small context doesn't error — the model emits `"..."` for every field** and writes it straight into the index. Context is sized from the prompt and placeholder payloads are rejected.
- **Thinking can't be turned off on the 30B**: with `think: false` it narrates its reasoning into the answer instead. So thinking stays on and the UI shows the librarian considering; the toggle is *show*, not *off*.
- **Grammar-constrained generation can run away and wedge Ollama** — three times in an hour, always on the same instruction-heavy pass. A `num_predict` cap and a short per-request timeout turned a 30-minute hang into a 22-second pass, because Ollama cancels a generation when its client disconnects.
- **A model will answer `true` while its own explanation says no**, even with the verdict field last — and will paraphrase the instruction back as the explanation. The reasoning wins over the verdict, and an instruction-shaped explanation discards it.

## Layout

```
library                 one-command launcher
src/library_agent/
  ingest/               extraction, section tree, chunking, dedup, bulk import
  reading/              tier 1 and tier 2 passes, versioned prompts
  retrieval/            hybrid search, router, reranked pipeline
  library/              clusters, citation graph, contradictions, taxonomy, shelving, cartridges
  ops/                  incidents and troubleshooting against the docs
  chat/                 citations, query rewriting, stances, streaming
  eval/                 question generation, recall@k harness
  api/  worker/  db/
web/index.html          the whole UI, one file, no build step
ops/                    launchd units, backup and restore
tests/
```

## Status

Complete through the planned scope. [`docs/DEVLOG.md`](docs/DEVLOG.md) is the phase-by-phase record of how it was built and what each measurement showed.
