# Model benchmarks

Which model should the library run on? The retrieval eval (`./library eval`) measures whether the right passage comes back, which is the same for every model. This measures the model's side of the work: how fast it is, how often it breaks the library's contracts, and how often it's right.

```sh
./library bench                                              # the chat model, every suite
./library bench --models qwen3:30b-a3b acme:qwen3:30b-a3b acme:gemma4-31b --runs 3
./library bench --suites raw,structured --models acme:some-new-model   # no library needed for these
./library bench --cases bench/results/our-questions.json --label ops-docs
```

A provider's model is named as in Settings › Models: `id:model`. The library must be running for `ask` and `write`. The run refuses to start while a generation is in flight on the shared GPU (a composition, say), since it would slow that down and be slowed by it; `--force` overrides.

## What it measures

| Suite | Measures | How |
|---|---|---|
| **raw** | speed of the model alone | a one-word reply (p50/p95); a ~250-word streamed answer (time to first token, time to the first word of the answer, output tokens/s, how much it reasons first); four streams at once (all complete? aggregate tokens/s) |
| **structured** | reliability of the library's own structured calls | Write's facet split, Tier 1's section reading and the conflict judge, each attempt counted: valid on the first try, valid within three, and why it failed (malformed JSON, placeholder values, prompt echoed) |
| | correctness of judgement | the conflict judge on four hand-made cases with known answers: two real disagreements, two look-alikes (two clusters' head nodes; two compatible properties of BM25) |
| **ask** | correctness, end to end | questions with known answers, each asked `--runs` times: the answer holds the terms a right answer must, cites the volume it should, every citation verifies, runs agree with each other; plus a question the library can't answer, which it should decline |
| | answer speed | time to the first word and to the whole answer, p50 and p95, including retrieval |
| **write** | a whole composition | one short document: coverage verdict, sections, citations verified, claims flagged by review, seconds per step |

Output tokens are estimated at four characters each, so every backend is measured with the same yardstick whether or not it reports usage.

## Questions

`cases.json` holds the default questions, on public papers (the Information Retrieval set). To benchmark on your own material, copy it into `results/` — which is gitignored — replace the questions, and pass `--cases`. Each question lists groups of terms a right answer must contain (any one term per group) and a piece of the title of the volume it should cite.

## Results

Each run writes `results/<date>-<label>/`:

- `report.md` — the tables, one column per model
- `raw.jsonl` — every measurement, one line each, for your own analysis
- `summary.json` — models, suites, library commit, corpus size, run time

`results/` is never committed: reports name your providers, and answers quote your library.

## Reading a report

- **First word** is what a person at the desk feels. A model that reasons first (qwen3) is slow to start and quick to finish; one that doesn't (gemma) starts at once.
- **Valid on the first try** below ~95% means the library is paying for retries on that call; a *placeholder values* failure is usually a context squeeze, *prompt echoed* a sampling one.
- **Citations verified** below 100% means the model cited passages it wasn't given; the library strips those, so the answer survives, but the model is less trustworthy with sources.
- **Runs agree** measures stability; **declines what it cannot know** measures honesty. A model that invents a source for the unanswerable question fails the most important test here.
- Ask timings include retrieval and reranking, which are the same for every model; compare differences, not absolutes.
