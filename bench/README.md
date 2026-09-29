# Model benchmarks

For anyone running the library: which model should it run on? `./library bench` measures the model's side of the work — how fast it is, how often it breaks the library's contracts, and how often it's right — **on your own shelf**, with your library running. The retrieval eval (`./library eval`) is the other half: whether the right passage comes back, which is the same for every model.

## Start

```sh
./library bench init                     # once: write questions from your library
./library bench                          # benchmark the chat model
./library bench --models qwen3:30b-a3b acme:qwen3:30b-a3b acme:gemma4-31b --runs 3
```

**`bench init`** samples passages across the volumes you hold and has the library's reader model write, for each, a question the passage answers and the key terms a right answer must contain. A term is kept only if it appears in the passage and isn't already in the question, so the model can't invent the answer key; the passage's volume is the one a right answer should cite. It writes `results/cases.json` — read it over, fix or delete a question, add synonyms to a term — and every later run uses it.

```sh
./library bench init --n 20                         # more questions (default 10)
./library bench init --cartridge "Ops Docs"         # only from one cartridge
./library bench init --shelf "Networking"           # only from one shelf and its sub-shelves
```

Only volumes at or below the classification ceiling that applies to the reader model are sampled, and the structured suite samples sections the same way for each model it tests — so a provider's model is never handed material above the remote ceiling.

A provider's model is named as in Settings › Models: `id:model`. `ask` and `write` go through the running library; `raw` and `structured` only need the models. A run refuses to start while a generation is in flight on a shared GPU (a composition, say); `--force` overrides.

## What it measures

| Suite | Measures | How |
|---|---|---|
| **raw** | speed of the model alone | a one-word reply (p50/p95); a ~250-word streamed answer (time to first token, time to the first word of the answer, output tokens/s, how much it reasons first); four streams at once (all complete? aggregate tokens/s) |
| **structured** | reliability of the library's own structured calls | Write's facet split, Tier 1 reading a section of your library, and the conflict judge — each attempt counted: valid on the first try, valid within three, and why not (malformed JSON, placeholder values, prompt echoed) |
| | correctness of judgement | the conflict judge on four built-in cases with known answers: two real disagreements, two look-alikes (two clusters' head nodes; two compatible properties of BM25) |
| **ask** | correctness, end to end | your questions, each asked `--runs` times: the answer holds the key terms, cites the volume it should, every citation verifies, runs agree with each other; plus a question no library can answer, which it should decline |
| | answer speed | time to the first word and to the whole answer, p50 and p95, including retrieval |
| **write** | a whole composition | one short document on your library's widest thread: coverage verdict, sections, citations verified, claims flagged by review, seconds per step |

Output tokens are estimated at four characters each, so every backend is measured with the same yardstick whether or not it reports usage.

## Your own questions

A question set is plain JSON:

```json
{
  "terms_required": 1.0,
  "questions": [
    {"question": "Which port does the backup agent listen on?",
     "terms": [["8443"], ["tls", "https"]],
     "volume": "Backup Agent Runbook"}
  ],
  "unanswerable": ["What did the library's owner have for breakfast on the day the collection was started?"],
  "write_brief": "How do our runbooks handle a failed backup, and where do they disagree?"
}
```

Each entry in `terms` is a group: the answer must contain any one term of the group. `terms_required` is the share of groups an answer must hold — 1.0 (all) for a hand-written set that lists what a right answer must say, 0.5 for a generated one, whose terms are exact strings from the passage that an answer may phrase otherwise. `volume` is a piece of the title the answer should cite. Pass a set with `--cases path.json`. `cases.example.json` is a hand-written example on public papers; it only fits a library that holds them.

## Results

Each run writes `results/<date>-<label>/`:

- `report.md` — the tables, one column per model
- `raw.jsonl` — every measurement, one line each, for your own analysis
- `summary.json` — models, suites, question set, library commit, corpus size, run time

`results/` is never committed: questions and answers quote your library, and reports name your providers.

## Reading a report

- **First word** is what a person at the desk feels. A model that reasons first is slow to start and quick to finish; one that doesn't starts at once.
- **Valid on the first try** below ~95% means the library is paying for retries on that call; *placeholder values* is usually a context squeeze, *prompt echoed* a sampling one.
- **Citations verified** below 100% means the model cited passages it wasn't given; the library strips those, so the answer survives, but the model is less careful with sources.
- **Runs agree** measures stability; **declines what it cannot know** measures honesty. A model that invents a source for the unanswerable question fails the most important test here.
- **The conflict judge's known cases** are four, which is enough to catch a model that systematically misses or invents disagreements, not to rank two good ones.
- Ask timings include retrieval and reranking, which are the same for every model; compare differences, not absolutes.
