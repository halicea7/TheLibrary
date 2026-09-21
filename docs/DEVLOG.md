# Library Agent

A local-only conversational interface to a personal library of papers, books, and docs.
Everything runs on this machine: Ollama for generation and embeddings, Postgres + pgvector
for storage and retrieval. No cloud calls.

## Design in one paragraph

Reading is a **quality tier, not a pipeline stage**. Tier 0 (extract → structure → chunk →
embed, no LLM) makes a document searchable in seconds. Tier 1 adds structural summaries,
entities, and categories in minutes. Tier 2 — the deep interpretive reflection pass — is
reserved for documents that earn it. This is what turns a 500-document backfill from ~260
GPU-hours into roughly one overnight.

## Status

- **Phase 0 — foundation**: done. 18 tables, pgvector 0.8.6, healthcheck green.
- **Phase 1 — ingest + Tier 0 + search + UI**: done.
- **Phase 2 — eval harness + retrieval tuning**: done.
- **Phase 3 — Tier 1 reading**: done.
- **Phase 4 — chat**: done.
- **Phase 5 — library layer**: done.
- **Phase 6 — Tier 2**: built and measured; kept as a reading artifact, cut from retrieval.
- **Phase 7 — UI, filtering, ops**: done.

## Setup

```sh
./scripts/bootstrap.sh      # once: pgvector, extensions, models, migrations
./library                   # every time: checks services, migrates, starts worker + API, opens the browser
```

`./library stop`, `restart`, and `status` do what they say. For a first load,
`./library import ~/papers ~/books --read` walks directories, skips anything already
shelved, and queues reading — faster than dragging hundreds of files through the browser,
though the drop zone walks folders too. Ctrl-C in the foreground closes
everything. `scripts/run.sh` remains for API-only development with `--reload`.

Requires Postgres 16 and Redis running locally, Ollama, and `uv`. Note that Homebrew's
pgvector only ships for postgresql@17/@18, so bootstrap builds it from source against
whatever `pg_config` you point at.

## Verify

```sh
uv run python -m library_agent.healthcheck   # services, models, vector round-trip
uv run pytest -q                             # ingestion invariants
```

## Models

| Role | Model | Notes |
|---|---|---|
| Reading (pinned) | `qwen3:30b-a3b` | Fixed corpus-wide: artifacts from different models drift |
| Chat — general | `qwen3:30b-a3b` | MoE, ~3B active |
| Chat — technical | `huihui_ai/qwen3-coder-abliterated` | Per-conversation toggle |
| Embeddings | `bge-m3` | 1024-dim, stored as `halfvec` |

Both chat models are ~19GB and **cannot be co-resident** on a 48GB machine; switching costs
a ~8-10s load. `keep_alive` is 30m rather than pinned, because this machine sits around
34GB used before any model loads.

## Notes

- Chunk ids are `uuid5(content_hash, char_span)`, so re-ingestion is idempotent and crash
  recovery is a replay rather than a cleanup.
- `embedding.owner_id` and `artifact.target_id` are polymorphic and carry no FK, so
  Postgres cannot cascade them. **Always delete documents via
  `library_agent.db.purge.delete_document`**, never a bare `DELETE`. `gc_orphan_vectors()`
  sweeps debris; `/api/health` reports the orphan count.
- In raw `text()` SQL, write `cast(:p as uuid)` — SQLAlchemy's bind-param lexer cannot
  handle `:p::uuid`.


## Retrieval evaluation

```sh
uv run python -m library_agent.eval.harness --generate 72   # regenerate + run the ladder
uv run python -m library_agent.eval.harness --config hybrid_rerank
```

Questions are generated *from* a known chunk, so the gold label comes free and recall@k
needs no hand-labelling. Every run is persisted to `eval_run` / `eval_result`.

**Two suites, because one query style cannot evaluate both halves of retrieval:**

- `retrieval` — paraphrased natural questions. The generator is explicitly told not to
  reuse distinctive phrasing, which is realistic for conversational use but strips out the
  exact-term signal lexical search exists to capture.
- `keyword` — short term-style queries ("ColBERT late interaction", "Hierarchical NSW").

Measuring only the first suite would have led to deleting the lexical half outright. It
loses on paraphrased questions at every weight and wins on keyword queries. `weight_lexical`
defaults to **0.2** as the compromise; revisit it once real query logs exist, since the
correct value depends entirely on how you actually type.

### Findings worth remembering

- `websearch_to_tsquery` and `plainto_tsquery` **AND every term**, so a natural-language
  question matched *zero* chunks and the lexical half silently contributed nothing. Use the
  `or_tsquery()` helper (added in migration `b1f3a9c0d2e4`) and let `ts_rank_cd` rank.
- Postgres full-text ranking has **no IDF term**, so common words drag in noise. This is
  why the lexical half needs down-weighting rather than equal fusion.
- The cross-encoder is the largest single win **on natural questions** (recall@1
  0.51 → 0.60, MRR 0.65 → 0.73) and a measurable **loss on keyword queries**
  (recall@1 0.42 → 0.39). Cross-encoders are trained on natural query/passage pairs; a
  2-6 word keyword string gives them too little signal and they override a correct dense
  ranking. So chat enables reranking (`CHAT_RETRIEVAL`) and the raw search box does not
  (`KEYWORD_RETRIEVAL`).
- Reranking costs ~65ms per pair on MPS and scales linearly, so depth is the only latency
  lever that matters. `rerank_depth=20` matches depth 30 on recall@1/MRR while lifting
  recall@10 from 0.940 to 0.985, at 1.2s vs 2.0s per query. Depth 100 costs 6s and buys
  nothing.
- The **document router is untestable at this corpus size**: with 12 documents and
  `router_top_documents=15` it selects everything and is a measured no-op. It is built for
  500 documents and should be re-evaluated there, not trusted on this evidence.


## Tier 1 reading

```sh
uv run arq library_agent.worker.tasks.WorkerSettings   # worker (run alongside the API)
curl -X POST localhost:8077/api/read/backfill          # queue every tier-0 document
curl -X POST localhost:8077/api/documents/{id}/read    # one document
curl localhost:8077/api/jobs                           # progress
```

Per document: an orientation card, one structured call per section, a document summary, and
category assignment — roughly 2 minutes for a 20-page paper. It then:

- **rewrites every chunk's context prefix** with the document's one-line orientation and
  re-embeds them, which is the contextual-retrieval payoff (a passage saying "the protocol
  requires a 32-byte nonce" becomes findable because its prefix names the protocol);
- **embeds the document summary**, which replaces the Tier 0 fingerprint as the router's
  vector.

Jobs yield to interactive chat between sections (never mid-call) via the Redis lease, and
record `yielded_reason` so the UI shows "waiting for chat" rather than looking stalled.

### Findings worth remembering

- **A too-small `num_ctx` does not error — the model emits schema-shaped placeholders.**
  Every field of the document summary came back as literally `"..."` at `num_ctx=8192`, and
  it was written straight into the index. `Ollama.generate/structured` now size the context
  from the prompt (`size_context`) and reject placeholder payloads (`is_placeholder`),
  retrying with double the context. This is the single nastiest failure mode found so far,
  because it is silent.
- Categories are steered at creation time by passing the canonical list into the tagging
  prompt. The periodic merge job stays a backstop rather than the primary mechanism.
- `artifact` rows carry `(model, prompt_version)`, so bumping a version in
  `settings().prompt_versions` marks only that artifact kind stale — see
  `reading.tier1.stale_documents`.


## Chat

`POST /api/chat` streams SSE: `conversation` → `meta` → `sources` → `token`* → `done`.
The UI at `/` has a Chat tab with a per-conversation model switch and a conversational toggle.

**Citations are structural, not self-reported.** The spec proposed letting the model tag
which parts of its answer are grounded. Local models are unreliable at that, so instead the
retrieved passages are numbered, the model emits `[n]`, and a post-pass resolves every
marker against the real sources — dropping any that don't resolve. `markers_resolved /
markers_emitted` is stored on every message as a mechanical groundedness metric. The
answering policy itself stays open per the spec: retrieval is context, not a cage, and
uncited prose is visibly the model's own reasoning.

### Latency: it was thinking mode all along

Slow first tokens were repeatedly attributed to memory pressure and prefill. The actual
cause: `chat_stream` never disabled thinking, so `qwen3:30b-a3b` generated a hidden
reasoning chain -- often thousands of characters -- before its first visible token. On a
real RAG prompt that is 30-100+ seconds of apparent silence.

It cannot simply be switched off. Measured on this build:

| | first token | hidden thinking | leaks into answer |
|---|---|---|---|
| `think: true` | 3.7s | 1,145 chars | no |
| `think: false` | 0.0s | 0 | **yes** — "Hmm, the user is asking me to…" |
| `think: false` + `/no_think` | 0.1s | 0 | **yes** |

With thinking off the model still reasons, it just writes the reasoning into the visible
answer. So thinking stays **on**, and the UI streams it as the librarian visibly
*considering* — faint, italic, replaced the moment real prose arrives. The wait is now
legible instead of a void. Prompt size (`chat_passages`, `chat_passage_chars`) still
matters, but it is the second-order term.

`think: true` combined with a JSON `format` schema returns an empty response, so
`structured()` keeps `think: false`; its guards (`is_placeholder`, `echoes_prompt`) exist
because that is exactly the mode in which reasoning leaks into fields.

### Memory is the binding constraint on this machine

With everything unloaded the machine sits at **35GB used / 12GB free**. A 19.3GB chat model
does not fit in 12GB, so it runs compressed: prefill degrades from ~175 tok/s to ~66 tok/s
and TTFT swings 27-44s. In that state measurements become meaningless — a 2.5GB model
benchmarked *slower* than the 19GB one.

Note that raising `iogpu.wired_limit_mb` would make this **worse**, not better: it lets
Metal wire more memory, and wired pages cannot be compressed or paged out. The real options
are a smaller chat model, fewer resident models, or closing other applications.


## Library layer

```sh
curl -X POST 'localhost:8077/api/library/rebuild?kinds=citations,clusters,contradictions'
curl localhost:8077/api/library/clusters      # cross-document themes
curl localhost:8077/api/library/graph         # citation edges + hubs
```

Three cross-document passes, surfaced in the UI's Library tab.

**Clustering runs on claims, not section summaries.** This was measured, and the difference
is the whole feature:

| clustered over | clusters | cross-document |
|---|---|---|
| section summaries | 29 | **2** |
| extracted claims | 119 | **23** |

A section summary ("this section describes RAPTOR's clustering algorithm") is irreducibly
about its own paper, so sections cluster with their siblings. A claim ("bigger models are
better") is atomic and comparable across documents. Claims are already extracted per
section at Tier 1, so this costs nothing extra.

**The citation graph needs no LLM at all** — reference sections are parsed with regex and
matched against corpus titles. Use `word_similarity()`, not `similarity()`: a reference
string is "Authors. Title. Venue Year", several times longer than a title, and plain
trigram similarity penalises that length gap so hard nothing matches (0.51 for an exact
title match versus 1.00 for `word_similarity`).

### Findings worth remembering

- **Constrained JSON generation emits properties in schema order, so a verdict field placed
  before its reasoning is committed to before the model has reasoned.** The contradiction
  detector returned `disagreement: false` while its own explanation said "they report
  opposite effects". Moving the boolean after the reasoning fields fixed all five test
  cases. `tests/test_reading.py::TestStructuredOutputOrdering` guards every schema against
  this.
- **Document-level context in chunk embeddings hurts.** Tier 1 originally appended the
  document's one-line orientation to every chunk prefix and re-embedded. Measured against
  the Phase 2 baseline this was flat-to-negative everywhere (keyword MRR -0.035). Appending
  a document-constant string adds the same component to every one of that document's
  vectors, making them *less* distinguishable. Anthropic's contextual retrieval generates
  context *per chunk*; a document-constant line is the degenerate case. Disabled via
  `orientation_in_chunk_prefix`; reverting restored the baseline exactly (+0.0000).
- The cluster significance filter is too permissive — "GPU Configuration" and "Hugging Face
  Integration" survive as themes when they are incidental. Prompt tuning, not architecture.


## Tier 2 — built, measured, and half cut

Tier 2 is the spec's original idea: the agent reads a document passage by passage and
writes *reflections* — what a thoughtful reader thinks, not a summary. Reflections are
genuinely good to read ("the abstract masks a deeper tension: it trades computational
complexity for contextual depth", "this subtly assumes current knowledge is static").

**It is far cheaper than the spec implied.** The spec called one LLM call per chunk; here a
whole section goes in one structured call, which amortises prefill and lets a reflection
reference its neighbours. Measured: **77s and 106s per paper** — not hours. The 810s outlier
was contention, not cost.

**But it does not improve retrieval, and that was measured twice.**

| suite | variant | r@1 | MRR |
|---|---|---|---|
| chunk-derived | tier1 / tier2 | 0.5278 / +0.0000 | 0.6325 / +0.0052 |
| chunk-derived + rerank | tier1 / tier2 | 0.6111 / +0.0000 | 0.6928 / −0.0069 |
| **interpretive** | tier1 / tier2 | 0.4889 / +0.0000 | 0.6403 / **−0.0221** |
| **interpretive** + rerank | tier1 / tier2 | 0.5778 / +0.0000 | 0.6743 / +0.0058 |

The first suite is biased — its questions are generated from chunk text, so it favours
literal passages. That is the same trap that nearly deleted lexical search in Phase 2, so a
second *interpretive* suite was generated (from chunks, never from reflections, which would
have been circular) asking about implications, assumptions and tensions. Reflections did not
help there either.

**Decision: `reflections_in_retrieval = False`.** Tier 2 stays as an opt-in reading pass
whose output you browse (`GET /api/documents/{id}/reflections`, and the "reflections" button
in the UI), promoted per document by `promotion_candidates()` — starred, frequently
retrieved, or a citation hub.

**What is still unmeasured:** whether including reflections in the *chat context* improves
answer quality. That needs an LLM-judge answer eval, which does not exist yet. The retrieval
eval cannot see it, and it would be wrong to claim either way.


## Running it as a service

```sh
./ops/install.sh      # launchd user agents for API + worker, survive reboot
./ops/uninstall.sh
./ops/backup.sh [dest]            # pg_dump + the content-addressed document store
./ops/backup.sh [dest] --no-vectors
```

Backup and **restore** are both verified: a 5.1MB dump plus 14MB of documents restores to
12 docs / 546 chunks / 1062 vectors, with vector search and the `or_tsquery` function
intact. Vectors are ~40% of the dump and are reproducible from the documents, but
regenerating them means re-reading the corpus, so they are included by default.

## Category filtering

Category chips scope both search and chat. A chunk is in scope if **either** its document
carries the category **or** the chunk itself does — the finer-grained half the spec asked
for, so one chapter of a general book stays findable on its own terms.

- `GET /api/search?q=…&categories=<id>,<id>`
- `POST /api/chat` with `category_ids`, stored on the conversation so every turn is scoped
  the same way.

Worth noting: the chips existed from Phase 3 but were **decorative** — they toggled UI
state and filtered nothing — until this phase. Verified by filtering to "Large Language
Models" and watching Dense Passage Retrieval correctly drop out of the results, and by a
nonexistent category id returning zero hits.


## UI

A single file, `web/index.html`, no build step. Three views — Ask, Find, Threads — over a
shelf of volumes. Two design decisions carry it:

**The citation apparatus.** An answer is a reading column; every `[n]` resolves into a
note in the margin beside the paragraph that cites it, with the volume, page, and section.
Hovering a marker lights its note and vice versa. This is the one thing a generic chat UI
cannot do, because this is the one chat that has mechanically verified page-level
provenance to show.

**Three accents, one meaning each, never traded:** madder red for a claim traced to your
shelf (the apparatus and nothing else — a flash of red always means provenance); verdigris
for system state (reading progress, selection, focus); amber for things wanting attention
(paused work, duplicates, sources that disagree). Serif for what the librarian says, sans
for the machinery, mono for the apparatus. Reading state is a filled square: empty is
*listed*, half is *read*, full is *annotated*.

Verified in Chrome: dark and light, the considering → answer → apparatus sequence, hover
linking, Find with page numbers and ranks, Threads with conflicts and citation edges.


## Remote Ollama

```sh
LIBRARY_OLLAMA_URL=http://your-box:11434     # every model call follows this
LIBRARY_OLLAMA_API_KEY=...                   # optional; sent as a Bearer header for a proxy
LIBRARY_KEEP_ALIVE=-1                        # pin models resident once the box has the RAM
```

Pull the models on the remote first (`bootstrap.sh` only pulls locally). Postgres, Redis and
the cross-encoder reranker stay on the API host — the reranker is in-process torch and was
never the constrained piece. The Redis lease that lets chat pre-empt background reading
still holds, since both still contend for the one remote instance.


## Follow-on: the murmur, and cartridges

**The murmur.** The model's hidden reasoning used to stream as a faint horizontal line in
the answer column. It now drifts upward *behind the shelf*, serif italic at very low
contrast, masked top and bottom: the library rifling through itself while it thinks. It
is always on, because it is the honest signal that a long first token is work rather than
a hang. Prose arrives in the answer column clean.

**Cartridges** are a portable slice of a library: a zip with a manifest, colour, icon,
confidentiality level and the data (`library/cartridge.py`). The design choice that made
it cheap: at the `readings` level every section becomes one synthetic chunk whose text is
that section's summary, with a deterministic id (`uuid5(content_hash, "reading:<order>")`),
and reflections are re-targeted at it. Nothing in retrieval, citation or the reader had
to learn a new source type — the passage *is* the reading. The document is flagged
`readings_only` so the UI says so.

Provenance is a column, not a namespace. `artifact.cartridge_id` says who wrote a note
(null is you); `cartridge_document.introduced` says whether an import created the
document or found it already shelved, which is what lets eject remove what it brought and
leave what was yours. Subjects merge by name through the existing taxonomy. The citation
rebuild used to wipe and re-extract from text; readings-only documents have no text, so it
now re-matches the raw references they shipped with instead.

The colour rule held: cartridge colours are identity, shown on rack spines, shelf dots and
a stripe beside the rubric rule on margin notes — never on a control. Scoping to one
cartridge is a room: the composer says *Ask Security's shelf* and the conversation stores
the scope so follow-ups stay in it.

Round-tripped on the live corpus: export *Information Retrieval* at `readings` (13
volumes, 146 reading-passages, 383 readings, 1.7 MB, no passage text in the zip); insert
onto the same shelf → 13 memberships, the 211 reflections aimed at reading-chunks
correctly skipped because the originals are here; delete one volume and insert v2 → it
returns readings-only with 8 reading-chunks; a scoped ask stays inside; eject removes only
what it introduced and leaves zero orphans.

Also from this stretch: `delete_document` now removes the stored file (the store is
content-addressed, one file per document, so it is safe), the importer skips files under
200 bytes, and the health check counts stray files. The lesson came from a payload
collection whose EICAR test string sat on disk after its document row was gone and set
off the endpoint agent.


## Follow-on: the shelf, properly

Feedback after living with it: thirty flat subjects, most security guides on three shelves
at once, and the whole thing read as a mess. The ask was *Cybersecurity › Network Scanning
› the document* -- two levels, no more, one place per volume.

Tags and shelving are now different things. `document_category` stays many-to-many for
filtering; `document.shelf_id` is the one sub-shelf a volume sits on; `category.parent_id`
gives the two levels. `library/shelving.py` runs the passes:

1. **Design** -- one call over every volume title, with thinking on. This was the hard
   part. Without thinking, qwen3:30b-a3b named sub-shelves after the tags it was handed
   (two seconds, lazy) or sliced the majority field into "Security Tools", "Security
   Concepts", "Security Methods" to fill the quota. Thinking plus a JSON schema *does*
   work -- the earlier "returns empty" finding was the reasoning eating the whole output
   budget -- so `Ollama.structured(think=True)` routes through `/api/chat` with room for
   both. A sanity gate rejects and retries the shapes it still reaches for: two top
   shelves sharing a significant word, activity names (Research, Preparation, References),
   schema-shaped filler (Top Shelf 1), a bay with one sub-shelf. `learning` was on the
   banned list for one embarrassing hour; Machine Learning failed every time.
2. **Fold** -- old tags map onto the new sub-shelves. An array with `minItems` guaranteed
   the *length* and the model filled it by repeating one subject; an object with one
   required enum property per tag makes coverage part of the grammar.
3. **Place** -- one short call per volume, reasoning first, sub-shelf from an enum, a new
   sub-shelf allowed under an existing top shelf. 0.5 s each.
4. **Even out** -- split any sub-shelf over 30 volumes from its own titles (catch-all
   names banned: "Web Attack Types" just recreates the lump), twice; then dissolve
   sub-shelves under 3 volumes into their siblings.

On the 170 read volumes: two top shelves (Cybersecurity 146, Machine Learning 24), 22
sub-shelves of 3–15 volumes, nothing unplaced, about seven minutes. Tier 1 places each
newly read volume; the shelf header has *reshelve*.

Also from this round: the murmur moved back into the answer column, in the empty space
under the question, masked by a radial bleed off the shelf's edge; and cartridges stand
on the rack as actual spines -- a slab in the cartridge's colour, name down the spine,
thickness proportional to the volumes it holds -- instead of chips. The mechanics had
been done; it looked like a filter.


## Follow-on: the nebula, and markdown

**The web.** `/api/library/web` returns a node per volume and three kinds of edge:
citations, shared threads, and nearest neighbours by document vector (three per volume,
cosine > 0.55) -- the last is what makes it dense enough to look like anything at 176
volumes, and what makes it thicken as the library grows. In the UI it is a force layout
in three dimensions on a plain canvas, projected with perspective and turning once every
two minutes; each volume carries an additive haze in its shelf's hue, which is what turns
a diagram into a nebula. Retrieval sends a verdigris ring out through it; the volumes an
answer drew on glow rubric and fade. Two bugs worth recording: nodes were born at
`canvas.width / 2` -- device pixels, so on a 2x display every volume spawned on the right
edge and never came back -- and the first tuning pinned everything to the boundary
because repulsion at spawn distance was unbounded. Cap the short-range force, cap the
velocity, scale the view to the cloud's real extent.

**Markdown.** A small renderer, inline pass first (escape, then code, bold, italic,
links, then the `[n]` citation marks so a mark is never taken for a link), then blocks:
fences, headings, quotes, tables, nested lists. The answer renderer hangs its margin notes
off each block instead of each paragraph. `.md` and `.rst` volumes render as written in
the reader; PDFs stay as text.

Logged for later: comparative questions retrieve from one volume (see TODO).


## Follow-on: settings, and the library troubleshooting itself

Asked for: a settings section with an error log where the same model helps fix things --
guiding, not doing -- using our own documentation as the reference, and suggesting a
GitHub issue with a ready description when that is the right move.

`ops/incidents.py`. Recording is fed from the API's unhandled-exception handler, failed
chat turns, failed jobs (one library pass failing no longer takes the others down) and a
logging handler on `library_agent.*` at ERROR. Messages are normalised (ids, paths,
numbers, quoted strings stripped) so a recurrence is a count, not a row.

Advice is one structured call, thinking on where the model has it: the incident and the
tail of its traceback, plus the most relevant paragraphs from `docs/TROUBLESHOOTING.md`
(new, written from what actually broke), the README, the devlog and the scripts, chosen
by keyword overlap weighted toward rare words. Output: diagnosis, likely cause from a
fixed set, steps with optional commands, the doc sections used, and an issue draft when
the cause is a defect. The technical model is used when configured; it answered a
tunnel-drop incident in ten seconds with the right section cited and the right three
commands, and a KeyError in our own code with a correct defect classification and a
usable issue body.

It also invented `./library cluster --dry-run` despite being told not to invent flags.
So the prompt is not trusted: every suggested command is checked against the commands
the documentation actually shows (fenced shell blocks, inline code, the launcher's usage
and case arms), and the UI marks the undocumented ones in amber. Home paths and anything
key-shaped are scrubbed from both the prompt and the issue text.

## Follow-on: Find in the nebula

The cloud stays up behind Find. When a search returns, the camera glides to the centroid
of the hit volumes and zooms to their spread (the projection now subtracts a camera
point before rotating, so the cloud keeps turning around whatever it is looking at); the
hits light in rubric; hovering a listed finding rings its own dot in verdigris and draws
the title beside it; clicking opens the volume. Clearing the query glides back out.

Two small things found on the way: `display:flex` on a class beat the `hidden`
attribute, so the desk bar leaked into Find -- a global `[hidden] { display:none
!important }` ends that class of bug; and a read job whose document was removed while it
waited its turn was recorded as an incident twice (explicit call plus the log handler).
An exception recorded explicitly is now marked so the handler skips it, and a document
that is gone by the time its job runs is not an incident at all.

**The tour.** "Be dramatic." When a search lands, the camera takes a beat on the whole
region, then dives on the first finding -- tight zoom, a spin impulse that decays so it
arrives with a swing, the dot ringed and pulsing, a leader line to the title set at 19px
serif with the shelf beneath -- and moves to the next every three seconds, the matching
row lit in the list. Hovering a row takes the wheel; leaving hands it back after a beat.
The camera eases at ~95% in half a second, which is fast enough to feel like a cut and
slow enough to read as motion.


## Follow-on: the cartridge as an object

Art, materials, dials, and "lock the design and have it carry over across instances."

`library/cartridge_design.py` holds the design (material preset, four dials, art source),
clamped on the way in and out; it travels in the manifest and is not editable after
insertion. Art ships as `art/label.png`; uploads and shipped art alike are re-encoded
through Pillow (metadata gone, 1024px cap, refused if it does not decode), same posture
as the SVG icons. The default label is generated: the cartridge's own constellation, its
volumes laid out by PCA of their document vectors and knitted by nearest neighbours in
the cartridge's colour -- deterministic, no image model.

`web/cartridge3d.js` is the object: a rounded slab from an extruded shape, grip ridges,
an edge connector with gold pins, a recessed label plate carrying a canvas texture
composed from the art and the name. Materials are MeshPhysicalMaterial presets:
transmission for clear and smoke (attenuation colour and distance carry the tint and
opacity dials), iridescence plus a cloud of additive sprites for glitter, metalness for
metallic. A see-through shell shows the constellation floating inside it -- the same
points as the label, spread through the plastic. Two things learned: three.js only
refracts the opaque pass, so the inner points must not be `transparent` (alphaTest keeps
the sprite edge); and refraction needs something behind it, so a rounded plate in the
page colour sits behind the object -- over the nebula it reads as a display case. One
vendored dependency, three r170, the first library the UI has used.

`./library import --cartridge` puts a folder on the rack directly, introducing its
volumes, which is how HackTricks arrived: 1,044 volumes in 564 s, a green translucent
cartridge with a thousand points inside.


**The rack as objects; light, pips, clearance.** The spine row is gone: the rack is a
three.js scene in the shelf column, every cartridge standing in a slotted base. Scoped
ones sit seated with a verdigris power light on; unscoped ones stand half out; toggling
eases the motion (~95% in a third of a second). Picking is a raycast; names and eject sit
beneath as HTML placed by projecting each slot. On the model: three level pips beside
the grip ridges lit by level, and a clearance band across the top of the label (open has
none; internal, confidential, restricted in slate, amber, rubric) with the marking
printed on it. Clearance is sealed with the design; `restricted` is also enforced at the
one place it can be: `resolve_selection` leaves out any volume that arrived only in a
restricted cartridge, so the receiving library cannot re-export it.


**One socket.** "You're overcomplicating it." The multi-slot rack became a single socket:
one cartridge in view, arrows or the wheel to step through the rest, a grid to jump. The
current one hangs above the socket; clicking drops it in and lights it. The motion is a
spring (stiffness, damping, a floor at the slot with a rebound) rather than a tween, so
the bounce is a real overshoot. Scoping became one room at a time, which is how it was
being used. Two things learned: MeshBasicMaterial goes through tone mapping like
everything else, so a backplate meant to match the page needs `toneMapped: false`; and
Chrome stops animation frames in a background tab, which makes a mid-motion screenshot
look like a bug.

**Clear backplates.** The refraction plate showed as an outline over the nebula. It only
needs to exist for three.js's offscreen transmission pass, so it now writes colour only
when a render target is bound (`onBeforeRender` checks `renderer.getRenderTarget()`) and
is invisible on screen: the glass still has something to bend, the page shows through
around it. Reflections on clear shells are dialled down so the tint reads without a lit
plate behind. Also found: the nebula's `extent` smoothing was a no-op from zero, so the
view was always scaled to the maximum radius -- unnoticed at 176 volumes, a blob at
1,219. It now starts from the first measurement and uses the 92nd-percentile distance,
so outliers cannot shrink the cloud.

**The models, properly.** "They look like they're from Roblox." Fair: primitives with a
paint job. Rebuilt the way the object is built: a front plate extruded with openings
cut out of it -- the grip grooves, three level pips, the power light, a screw -- so
every recess has chamfered walls; a back plate; a rim joining them; a PCB inside with a
few components and an edge connector of individual gold contacts; the label as a sticker
on the outside of the front face with its own raised edge, never behind the glass. The
plastic has a fine grain from a procedural normal map, a clearcoat, and a sheen on solid
colours. Lighting is a real studio HDRI (Poly Haven, CC0) through three's RGBELoader,
prefiltered once per page, with a soft-shadow key light and a contact shadow on the
socket. Two bugs on the way: `ShapeGeometry` needs a `Shape`, not a `Path`, for the
groove floors; and a label plane coplanar with the top of its own sticker edge z-fights
into stripes. Through a clear shell you now see the board, and the constellation floating
between it and the front.

## Follow-on: other tools at the desk

Someone else's tool wants to talk to the documentation through the library. Two doors,
one rule. The rule (`api/auth.py`): loopback is the UI and needs nothing, as it always
did; anything from off the machine must carry a bearer token from `LIBRARY_API_TOKENS`,
and with none configured it is refused. Bind stays loopback until `LIBRARY_BIND` says
otherwise -- two deliberate steps to open it, none to keep it closed.

Door one is `/api/v1` (`api/routes/v1.py`): plain JSON, no streaming. `ask` collects a
whole chat turn -- the same retrieval, generation and citation verification the UI gets
-- and returns the answer with only the citations that survived verification. Rooms and
shelves are addressed by name, since a tool should be able to say "our docs" without
learning our ids. Door two is MCP (`mcp_server.py`, mcp 2.x): four tools over stdio or
streamable HTTP, a thin client of door one, so the one-generation-at-a-time lease and the
door itself stay in one place. Checked with the SDK's own client over stdio, and the
JSON ask against HackTricks came back on the technical model with 7 of 8 citations
resolved -- the eighth was the model citing a number it had not been given, stripped
before the caller saw it, which is the whole point of the apparatus.


## Follow-on: the library writes

"Can it compose documents we can save?" `chat/compose.py`. A brief goes in; the
librarian plans an outline with thinking on (title, sections, and for each a *search
query*, not a question -- the plan is also the retrieval plan), then writes each section
against passages retrieved for that section. Citation numbers run across the document (a
passage seen again keeps its number), each section is verified against only the passages
it was given, and a references list of what was actually cited closes it. Streamed to
the Write tab section by section with the murmur running; saved as `.md`; printed via a
print stylesheet; or shelved, at which point the document is ingested like any dropped
file and becomes a volume the library can read. `POST /api/v1/compose` returns the same
thing as JSON.

First run on the Machine Learning shelf: a four-section reading guide, 33 of 33
citations verified. Two things fixed on the way: a local `sources` in the UI shadowed the
global the apparatus reads (no margin notes until it was renamed), and the model opens a
section with its own heading no matter how it is told not to, so the first line is
dropped when it matches.


## Follow-on: consistency, measured

Retrieval re-evaluated on the 1,219-volume corpus with 57 freshly generated questions
(the old suite had been cascade-deleted with its gold documents): hybrid RRF MRR 0.71,
recall@10 0.91, against 0.65 / 0.94 two days earlier at 176 volumes. A sevenfold larger
corpus cost almost nothing; lexical alone fell to MRR 0.17 on the security material,
dense held at 0.69, fusion on top 0.71. The reranker did not help these keyword-shaped
questions (0.69) and the router hurt (0.70, recall@10 0.86): both findings from Phase 2
hold at scale.

`scripts/consistency.py` asks questions with known answers three times each without
memory and scores volume cited, required terms present, citations verified, runs agree,
plus one unknowable question that must be declined without a citation. It could not run
this session: partway through, the remote Ollama wedged -- HTTP answering, both models
resident, no generation returning even with every client disconnected -- which is the
failure `docs/TROUBLESHOOTING.md` describes and the one thing this side cannot fix.


Ollama restarted on the remote; the consistency script ran on both chat models, three
runs each over five known-answer questions and one unknowable one. Every known answer
correct, every expected volume cited, 162 of 162 citations verified across both models,
every question consistent across its runs. The technical model is twice as fast (2.6 s
median against 5.0) and used more sources per answer. The unknowable question was
declined every time; the general model's decline cited the one breakfast in the library
(a squirrel's, in a T5 example) and said it did not apply -- my script first scored that
as suspect for having a citation, which was the script's mistake, not the librarian's.


## Follow-on: running it for others

"I just want the engine to be as consistent as possible since other users are going to
start using this." The content was already consistent (measured above); what a shared
engine also needs is to fail plainly. `llm/liveness.py`: a one-token probe every two
minutes, because today's wedge passed the reachability check -- `/api/tags` answered,
both models resident, nothing generated -- and every caller would have hung for the full
timeout. While the probe fails, `ask` and `compose` return 503 with the reason at once.
A gate: a small global limit, one in-flight generation per token (the person at the desk
is exempt), a queue timeout into 429 with Retry-After. A deadline on the JSON `ask`
(504). A cooler default temperature for API callers, and a `deterministic` flag (seed,
temperature 0) that is honestly best-effort: through the full path four of five runs were
byte-identical, the odd one the first call after a model swap, and Ollama on the raw
generate showed the same two-of-three -- GPU batching, not us. The "passages are data"
line in the system prompt is now always on, since documents come from other people.


## Follow-on: effort

"Can we toggle effort levels like many harnesses do?" Not in the model: qwen3 thinks or
it does not, and off leaks the reasoning. So `chat/effort.py` makes effort the work
around the model. quick: three passages, no reranker, no rewrite, 8k context. normal:
today. deep: the question is first split into two to four searches (a small structured
call, falling back to the question itself), each retrieved, the union sorted by score
and cut to ten, reranked at depth 40, 32k context. On the comparative question the TODO
had flagged -- SSTI versus SQL injection -- quick and normal cited one side; deep cited
both, from four volumes, ten of ten markers resolved, in 23 s against 2.5 and 10. The
dial is in the tab bar, remembered, and on the JSON and MCP asks.


**Batch reading from the shelf.** Two buttons under the shelf header, counted from what
is in view: *read the N unread*, *annotate the N read*. Scoped like everything else --
a seated cartridge, a chosen subject -- two-step armed with an estimate of the hours,
and idempotent: volumes already queued or running are skipped. First press queued 78
where the button had promised 12, because the endpoint counted tag membership expanded
to child shelves while the button counted shelf placement; the queue was cleared and the
server now uses the button's rule (shelved under it, or tagged with exactly it).


## Follow-on: making the engine repeatable

Three TODO items about consistency, worked through in order.

**Normal mode citing one side.** At k=5 a question that spans two volumes could come back
with five passages from one of them, so the answer cited one side of a comparison. After
fusion the pipeline now caps passages per document (two in chat, three in deep) and
back-fills from the rest of the ranking, so the second volume reaches the model. The
cap is a reorder of what fusion already ranked, not a new retrieval source; the eval
harness ladder gained `hybrid_rerank_diverse` so the cost is measured, and on the
57-question set it is neutral (the questions there are single-volume).

**Contradictions that changed every rebuild.** Identical runs at temperature 0.1
returned 0, 1, 3, 4 and 7 findings: the borderline clusters are coin flips. Each cluster
is now judged up to three times at temperature 0 with seeds 1, 2, 3, and a finding
needs two votes; a first clean *no* ends it early, since most clusters are clean, so the
pass costs about a third more rather than three times. Cluster significance got the
same treatment the cheap way: a cluster the first vote would hide gets a second vote
with another seed and is shown if they disagree. `seed` is now plumbed through
`Ollama.generate` and `structured`.

**Reflections in the answer.** Retrieval said twice that reflections do not help find
passages. Whether they help *write* the answer is a different question, and
`scripts/reflections_ab.py` asks it: questions generated from annotated passages,
retrieval done once and shared, the same passages sent twice -- bare, and with the
library's note beneath each -- judged blind by the reader model on groundedness and
usefulness, with citation validity alongside as the metric that needs no judge.
`render_context` takes an optional `reflections` map to make the B arm possible; nothing
uses it in the chat path until the A/B says it should.

**Figures.** "Is there any way to include images, or would that bloat?" It does not
have to. Extraction already finds the drawing and image regions on a page (to keep
their labels out of the prose), so `ingest/figures.py` reads the same regions the other
way: each one large enough to be a figure -- between 3% and 85% of the page, not a rule
or a logo -- becomes a figure, with the nearest caption block beneath it, and the reader
renders it from the stored original at 2x on first view. No table, no migration, no
extraction at ingest; the renders and index are a cache under `~/.library-agent/figures/`
that the document's deletion removes. Cartridges came free: `full` ships the original so
the receiver's reader finds the same figures, `readings` and `catalogue` ship none. On
BERT, ResNet and DPR the captions attached to the right figures first try; the clip
stops above the caption so the text stays selectable beneath the image. Markdown images
are deliberately not fetched -- a shelved file should not phone the site it came from.
Reading figures with a vision model into caption chunks stays open; it needs a model
pulled on the GH.

**A shelf that collected the wrong things.** Twelve volumes under *Machine Learning ›
Information Retrieval* turned out to be Git, Mercurial, Twitter, MSSQL Server -- one-page
stubs from the payload collection. Two causes, both in what the placer was told. A stub
with no document summary was placed from its title alone, and "Git" from its title alone
"aligns with Information Retrieval"; the placer now offers the volume's opening text when
there is no summary. And the volume's own current shelf was passed back to it as a tag,
so a wrong placement was its own evidence on every re-placement; the current shelf is now
excluded from the tags. A dry run on the four worst placed three of them correctly
(Attack Techniques, Database Injection, Security Misconfigurations). `reshelve` takes a
`category_id` to re-place one shelf's volumes without rebuilding the taxonomy, and the
shelf header grows a *re-shelve these N* button when one shelf is chosen.

**Reflections in the answer: no.** The A/B ran to 14 of 20 questions before the model
wedged (HTTP up, no generation; the liveness probe caught it and the masthead said so).
Enough to decide: plain preferred 7, notes 4, tie 3; grounded 4.29 plain versus 3.57 with
notes; useful 4.43 versus 4.29; citation validity 1.00 both. The pattern in the judge's
reasoning was consistent -- with a note beneath a passage the answer asserts the note's
reading as if the passage said it, and the passage often did not. So the third suite
agrees with the first two: reflections are for the reader in the margin, not for the
model in the context. `render_context(reflections=)` stays as the tested hook; nothing
in the chat path calls it.

**Contradiction stability, measured.** Every cluster the rebuild had flagged (25) plus 25
random clean ones, each judged three times by `judge_cluster` on a quiet model: **50 of 50
agreed with themselves** across the three runs, against 0–7 findings per run before. The
honest footnote: 6 of the 25 flagged came back *no* all three times now. Those were flagged
during a rebuild that ran while another process was also generating -- and the two of them
together drove Ollama to a 500 -- so borderline verdicts under GPU contention are not the
verdicts of a quiet pass. Majority-of-three removes the coin flip within a pass; it does
not make the model the same machine under load as at rest. The rebuild runs alone on the
worker, which is the normal case.


## Follow-on: the highlighter, and Threads you can actually read

"Keyword highlighting for easier read on Find and Threads; Threads is cluttered." What
Threads was: 209 clusters as a flat list of equal cards (125 of them two-volume pairs),
21 conflicts on top as paragraphs, then eighteen rows of truncated `A → B` citations --
and none of it scoped by the chips, so a security reader scrolled past the ML threads.

**Find.** The query's words lit in the passage, word-prefix so *inject* lights
*injection*, a soft amber wash (amber is the attention colour and reads as a marker).
The catch is dense retrieval: the best hit often has none of your words. So `/api/search`
now embeds the sentences of the hits in one call (`retrieval/lift.py`) and returns each
hit's sentence nearest the query; the UI underlines it, and leads the clamped excerpt
with it when it would otherwise be cut off. About a second of extra latency on the
tunnel, paid once per search.

**Threads.** Ranked by reach; four or more volumes gets a card with the summary clamped
to three lines (click opens it), fewer folds to a `<details>` line. Conflicts capped at
five with an *all N* toggle. A filter box that lights the word wherever it appears --
the summaries' own label words are *not* lit; that was tried and it made the SSTI cards
a wall of amber. The chips scope Threads through a member-volume clause on
`/api/library/clusters` and `/contradictions`. The nebula now stays up behind Threads:
hover lights a thread's volumes, click dives on them. The citation rows are gone; the
nebula draws those edges.

**Conflicts as quotes.** The judge had never been told which document each claim came
from, so its explanations paraphrased -- and on the stub pages invented ("one source
claims DNS brute-forcing is insecure" from a claim that said "References section is
included"). The cluster summary now keeps `claim_sources` parallel to `claims`, the
judge sees `- [Title] claim`, and its schema gains `claim_a/source_a/claim_b/source_b`.
A quoted claim is kept only if it is found among the claims the model was given; the
UI shows the pair side by side with their shared words lit. A conflict whose explanation
is that pages say their content moved to different URLs is the mirrored wiki's stubs
talking, and is now junk. Needs a threads rebuild to take effect.


## Follow-on: the day room

"Light mode looks abhorrent." It did: `#fff` panels, the nebula a grey smudge, the socket
a black slab on white. Nothing in it was ours. The brief that followed -- Greco-Roman,
gold, marble, columns, "alluding to the Library of Alexandria" -- resolved, after a talk,
into *no marble texture, just the inspiration*.

**The palette.** Papyrus, not paper: `#ece4d3` ground, ink `#2a2118`, rules the colour
of a reed pen's edge. The pigments keep their meanings in both rooms -- rubric is still
"from the shelf", verdigris still "system" (it is bronze gone green, which fits), amber
still "attention" (ochre by day). One new token, `--gilt`, is chrome only: the lintel
fillet under the masthead, the frieze hairline that runs out from every section label,
the chosen tab, the progress bar. Gold on edges and letters, never as fill. The tabs and
labels moved to the serif with inscription tracking.

**The chart by day.** A nebula multiplies into white as grey. By day the haze is nearly
off (alpha .022 against .17 at night), the edges carry the drawing at twice their weight,
and the points are sepia ink at 28% lightness. It reads as a star chart engraved on the
page rather than a cloud sitting on it. Lit volumes glow at a quarter of the night alpha,
since multiply makes rubric black. And Threads no longer lights every volume on render --
that turned the whole chart red -- lighting is per thread, on hover and click.

**The pedestal.** Real geometry, not a texture: two stepped slabs, a drum with twenty
flutes (a scalloped section extruded upward), an echinus lathe, an abacus slab, a bronze
frame around the slot and a dark mouth beneath it, gilt fillets at the lip and the foot
and a gilt band along the abacus face. Stone follows the room -- limestone by day, basalt
by night -- re-stoned the moment the theme flips, since the rack is idle when nobody is
touching it. First cut shadowed a local named `frame` and fell back to the 2D spines
without a word beyond a console warning; worth a louder failure some day. The cartridge
stays the object it was: its design is sealed, and a thing does not change colour when
you carry it into another room.

**Second pass.** "Nebula still looks kinda dark on light mode; can dark mode be darker,
almost black rather than that blueish dark gray." Night room to `#0a0b0d`, neutral
rather than blue, rules and text greys re-derived from it; basalt a shade darker to sit
in it. Day chart: ink lightened to `92,72,48`, points at 40% lightness and 42%
saturation so the shelves read as coloured inks, edge alphas roughly halved, haze to
.014. It is a drawing now, not a stain. And with the `[Title]` prefix peeled, all 13
conflicts carry their quoted pair.

**Screenshots and the README, reworked.** Every screenshot retaken on the current build:
the night Ask and the day Ask-in-a-room as the pair of heroes, an answer with its
apparatus, Find with the highlighter and the lift, Threads in the day room with the
quoted pairs, the reader on BERT with Figure 1 and a reflection beside the passage, the
pedestal in both rooms as one composite. The README's front half was rewritten into
*What it is → Using it → The desk → How it works*, with each image beside the feature
it shows, the duplicated nebula paragraph gone, and the newer mechanics (per-document
cap, placement evidence, quoted conflicts, figures from the same regions extraction
already finds) folded into *How it works*. The reference sections were kept as they
were. Found on the way: the figure card in the night room was a white box with an
invisible caption; it takes the panel colour now, with the render on its own white paper.


## Follow-on: HTML on the shelf

"My buddy wants a demo; I'm thinking of exporting their entire Confluence base -- HTML,
PDF, CSV or XML?" HTML, and rather than a pandoc step (not installed here anyway), the
importer takes `.html` itself. `ingest/html.py` finds the page body by a list of
selectors -- Confluence's `#main-content`, MediaWiki's `#mw-content-text`, `article`,
`main`, `[role=main]`, then `body` -- strips the chrome that lives inside it
(breadcrumbs, page metadata, the attachments and comments blocks, footers, nav, scripts),
converts with markdownify (ATX headings, fenced code with the language taken from
Confluence's `brush:` parameter or a `language-` class, tables as tables), and puts the
page title first as the H1, with the space name split off ("Space : Page"). From there it
is a Markdown volume: `MARKDOWN_LIKE` in extract.py, kind `doc`, rendered as written in
the reader. Tested on a Confluence-shaped page and through the upload route.

**Editing a cartridge made here.** "I want to edit that cartridge in the UI, don't lock
it -- add art and whatnot." The seal was for cartridges that *arrive*: their maker set
them. A cartridge made on this machine from a folder (`made_by == "import"`) has its
maker here, so it is `editable`: `PATCH /api/cartridges/{id}` takes name, colour, icon,
design and art (or `clear_art`), a foreign cartridge answers 403. The rack plaque grows
*edit*; the make panel opens filled from the cartridge with its volumes as the fixed
selection, *save to the rack* patches it, and *make it* exports with `as_cartridge`, which
keeps the id and moves the version on so the receiver upgrades in place. Along the way:
the jobs listing now puts live jobs first, since a queue of 391 had hidden the one in
hand behind the limit, and the UI counts the whole queue.

**Seven rebuilds for one import.** The debounce was a ten-minute reservation key: the
first read to finish in a quiet window scheduled a rebuild ten minutes out. An import
whose reads run for eighty minutes crosses eight windows, so it scheduled a rebuild per
window -- seven forty-minute passes queued behind each other, each blocking the reads
still waiting. Now a scheduled rebuild carries the time it was asked for; when it comes
due it first checks the shelf: reads still queued, it steps back into the queue behind
them; a rebuild already started since it was asked for, it is covered and returns. The
duplicates in Redis were dropped by hand this once.

**Restricted binds the receiver.** The first automatic export of the wiki cartridge
came back *nothing selected*: while it read, its maker had set the clearance to
restricted from the panel, and `resolve_selection` excluded every volume introduced by a
restricted cartridge -- a rule written for cartridges that arrive. A cartridge made here
is the maker's own; the mark is for whoever receives it. The exclusion now applies only
to cartridges not made on this machine. 391 pages, 387 read, 15 MB, version 3.


## Follow-on: the model sees the figures

`qwen2.5vl` was pulled on the GH. `reading/figures.py` sends each figure the extractor
already finds (rendered at 2x) to it with the page's caption and asks for one plain
paragraph, at most 120 words: what it shows, the axes, the trend, the numbers. The
answer becomes a chunk of kind `figure` -- a new column, default `text` -- with the page
it sits on, filed under the section that holds it, sorted after every text chunk, and
embedded like any passage. So nothing downstream changes: Find lights it, Ask cites it by
page, a cartridge ships it (`kind` joined the chunk columns), and the reader keeps it out
of the running text and shows it under the figure instead, below a rule, in the sans so
it does not read as the page's own caption. Five seconds a figure; BERT's five took ten.
On "how is BERT fine-tuned on different tasks, diagram", Figure 1's passage was the
third hit. Runs at the end of Tier 1 for PDFs, its failure never the reading's; a
`figures` job and `POST /api/read/figures` backfill volumes read before it existed.
The four RTS pages the model refused twice read fine when called by hand, so they were
queued a third time; HackTricks is being fed to the worker in batches of 150.

**The tour.** "Reshelve would be easily missed by someone -- like when programs highlight
sections and explain step by step, in the UI, a soft lock." Thirteen steps: the drop
zone, read/annotate, reshelve (the one that prompted it), the rack, each tab, effort,
stance, chips and rooms, the two rooms. A fixed spot element with a 100vmax box-shadow
in the room's own background at 72% is the dimming and the cutout at once, a gilt
hairline rings the target, a card sits to the right when there is room, else below, else
above. Steps that need a tab switch it; steps whose target is not on screen are passed
over, so an empty library gets a shorter tour. Esc, arrows and Enter drive it; clicking
the dimmed page advances rather than escaping, since that is what a first-timer does.
First visit only (`tour.done` in localStorage); *show the tour* in Settings brings it
back.

**When a reshelve is due.** "It should detect when a reshelve is needed, and outline
the button with a glowing tracer -- not a full glowing box, a tracer that travels."
Detection needed one fact the library never kept: when the shelves were designed and
on how many read volumes. A `library_meta` table holds it (`record_design` in
`build_taxonomy`). `shelf_health` then says due when there are no top shelves, when
read volumes sit unplaced, or when the read collection has grown by a quarter (and at
least twenty) past the design; a reshelve already in hand silences it. The tracer is a
conic gradient with one bright arc, masked to a hairline border with `mask-composite:
exclude`, its start angle a registered `@property` so it can animate -- one point of
gilt light going round the button every 2.6 s. Reduced motion gets a still ring.

**Materials, revisited.** "Clear, smoke and glitter look the same." They shared one
transmission branch with small offsets; the folder of Godot materials left in `dump`
was an encrypted export, unreadable, but its names said what the set should be. Six
now, each defined by what light does: clear passes it straight (transmission near 1,
roughness near 0, hard clearcoat); frosted, new, scatters it at the surface (rough
transmission, thin); smoke absorbs it in the body (dark attenuation, short distance);
glitter keeps its colour and is full of flakes, with a speckle roughness map so the
surface itself catches light in points. The flakes had been invisible all along for a
reason worth writing down: three.js draws only opaque objects into the buffer a
transmissive shell looks through, so additive, transparent points inside the body never
appeared. They are opaque points now, half of them just under the front face.

**Pause.** One Redis key, `library:paused`, checked by the same gate that already
steps aside for chat: between sections of a read, between clusters of a rebuild (the
library passes now take the gate too), and at the door of every job before it starts.
The job row says *paused* while it waits; a rebuild that finds the flag at its door
defers itself two minutes rather than hold a row open. `POST /api/jobs/pause?paused=`,
`GET /api/jobs/state`, and the *In hand* header grows the control -- amber *resume*
while paused. Found on the way: the "already ran" check for duplicate rebuilds was
keyed on *start* time, so a rebuild cut off by a worker restart skipped its own retry;
it is keyed on completion now. And the batch feeder tops the queue up at ten remaining
instead of letting it drain, so a batched import gets one rebuild at the end rather than
one per batch.

**A pause of a night.** The first long pause found the hole: the paused job waited
inside the gate, and after three hours arq's job timeout killed it -- one document
failed, its row stuck on *yielded*. A pause longer than ten minutes now sends the job
back to the queue (`Retry`, deferred five minutes) instead of holding it; what the read
had already written stays, and `max_tries` is raised so a long pause is many small
retries rather than a failure. Genuine exceptions are never retried by arq, so the
limit only bounds pausing.

**Materials from the decompiled set -- a contribution.** The Godot materials in `dump`
were decompiled by another assistant (GPT Astra) working directly in this checkout,
which ported the casing and label shaders into `web/cartridge3d.js` through
`onBeforeCompile` hooks on the physical material: a brushed shell surface, and label
*finishes* -- paper, gloss, holographic, prism, gold, chrome -- with a strength dial,
sealed into the design like everything else (`LABEL_FINISHES`, clamped, tested). A
second pass made the make-panel preview draggable (pointer capture, pitch clamped) and
narrowed the stage. A note for the record: both passes landed in this checkout while
my own commits were being made with `git add -A`, so they went to GitHub under the
"pause" commits of 04:02 and 13:20 rather than their own message. This entry is the
credit those commits should have carried.


## Follow-on: scope

"'SSH Access Configuration' is a conflict because it's different SSH addresses, not
because the information conflicts -- it's simply a different host." Every conflict the
library showed on the wiki was that shape: two clusters' head nodes, a key requirement
and a key-plus-2FA requirement, the same instruction in past and future tense. Two
causes. The judge never saw a claim's scope -- claims arrived as bare sentences under a
title -- and the prompt said a clash "is a conflict even when the sources study
different systems", written for papers, where a method's result should carry across
setups, and exactly wrong for documentation, where a claim is about its document's
host. Now each claim carries its document's orientation line (`[OSG Setup — OSG
integration guide for the Viper cluster] Head node is login01…`), the prompt says a
claim is about the thing its document is about and lists what is not a conflict, and
the schema asks for the one `subject` both claims share -- optional, since an empty
required string trips the placeholder guard (the first run lost every verdict that
way), enforced after the fact: no shared subject, no conflict. Re-judging the 99
flagged clusters by hand: 7 survive, each naming a specific thing -- the mount state of
one filesystem, the log path on one system, rsyslog's direction to one host. The
end-of-import rebuild applies it to the shelf.

**The rack kept the old label.** Changing a cartridge's colour recoloured the make
panel's preview but not the rack's: the panel renders the constellation fresh through
`/art/preview?colour=`, while the rack fetches `/art`, which draws the constellation
*once* in the cartridge's colour and stores it -- and then serves that file forever,
with no cache headers, so even a regenerated one came back from the browser's cache
under the same URL. Three small changes: a colour change on a generated label drops the
stored render; `/art` answers `Cache-Control: no-cache`; the rack and the inspect card
put the colour in the art URL so the label texture's cache key changes with it.


## Follow-on: the hosted demo

"Could it be based off our repo, so we don't update it manually every time the UI
changes?" It is the repo: a Pages workflow uploads `web/` untouched. The one addition
to the page is a two-line bootstrap at the top of `index.html`: on a `github.io` host,
or with `?demo`, it document.writes `demo.js` before the page's own script. That file
replaces `window.fetch` for `/api/...` with placeholder material of the right shape --
seventeen made-up volumes on two top shelves, two cartridges, a nebula of a hundred and
sixty points, findings for a few queries, two disagreements with quoted pairs, one
answer that streams as the server would (thinking, sources, tokens, done) and one
composed document -- and swaps the first-visit tour for eleven longer steps that
explain how each part works while doing it: the Ask step submits the question and the
visitor watches it think. Writes answer with a note that this is the demo. Asset paths
went relative (`./vendor/`, `./cartridge3d.js`) so the same files serve from `/`
locally and from `/TheLibrary/` on Pages. Nothing real is in it, by design.


## Follow-on: passages in hand

"Could findings from Find be thrown as context into Ask?" — and *lead* it, not replace
it, for Find and Threads both. A `hold` toggle on each Find hit and *hold both* under a
disagreement (the passage behind each claim, found by searching the claim inside its
own volume; `/api/search` takes `documents=` for that, and an explicit document scope
now wins over the router's top-up). Held chunk ids travel as `pinned_chunk_ids` on
`/api/chat` and `passages` on `/api/v1/ask`; `hits_for_chunks` turns them into hits
scored above anything fusion produces, they take the head of the list and retrieval
fills the rest of the passage budget. `Source.held` rides the `sources` event and the
note in the margin says *held*, so the answer shows which evidence was chosen and which
found. A tray under the composer lists what is in hand across tabs and follow-ups. One
bug on the way: titles with quotes broke an inline handler that inlined them; the
button carries data attributes now.

**The cartridge, remade (contribution).** A second pass by GPT Astra on
`web/cartridge3d.js`: a molded-shell construction with two rim halves and an assembly
gap closed by a recessed tongue, a continuous molded floor under the label pocket and
grip depressions, the label as a rounded pocket with its own texture (a fade under the
wordmark, the name wrapped), level pips labelled Catalogue / Readings / Full, lower
fasteners in real openings with bezels and a cross recess, a recessed rear service
panel with engraved marks, and traces and parts on the board so a clear shell has
something to show. Reviewed: nothing fetched, the design contract untouched, disposal
covers the new geometry. It deploys to the hosted demo on push like any UI change.

**Booting.** "When you insert the cartridge, the level pips should turn green one by
one, as if loading." First cut turned the LED on at the click, then off, then the pips
off, then lit them in sequence -- "kinda awkward", and it was: nothing should go dark on
the way in. Now the pips are always lit (the cartridge's colour at rest); seating
*arms* the cartridge, so the power light waits; and the first frame it sits still in
the socket after the bounce starts the boot: the lit pips *turn*, one level at a time
-- catalogue, readings, full, 380 ms apart, each with a flash -- to a green that pops
(`#37e07a`, deliberately louder than the verdigris the UI uses for system state), and
only then does the power light come on. Seated, they stay green; lifted, back to
colour. A cartridge arriving already seated arms rather than lights, so opening the
page runs the sequence once.

**The drawer.** "*all* should be a floating window with a grid of cartridges and info
about them -- make it look good but useful." A gilt-lined window over the desk: cards
with the label art, colour band, level pips, clearance banner, seated dot, count and
origin; a side panel with the chosen cartridge turning live (one more `mount`, drag to
turn, disposed on close) above its particulars -- clearance, material and label finish,
origin, when, embedding model -- and the acts: seat or lift, show on the rack, edit for
one made here. Double-click seats. Esc or the dim closes. The old two-column list of
names is gone.


## Follow-on: genre

"How does it handle discernment around import types? RTS is internal documentation and
should be digested with that in mind; a box of emails is a different data type." It
did not, beyond a `document_kind` guessed at orientation and read by nothing. Now
`reading/genre.py`: eleven genres, each with a rule for what a *claim* is (a runbook's
claims name the host they apply to; correspondence is `who -> whom (date): what was
asserted`, never merged across senders), a phrasing for a theme's entry, and a rule for
what a conflict is on top of the scope rule. `document.genre` is set at orientation
unless the maker set it first -- a cartridge has a genre too (`PATCH`, or `import
--genre`), applied to all its volumes -- and it ships in cartridges. The cluster
summary keeps the dominant genre with its claims so the judge reads it without another
join. Existing orientation cards were folded into the column by the migration (1,206
documentation, 130 paper, 83 report). Volumes already read keep their readings until
read again; threads pick the genre up at the next rebuild. Email ingestion -- headers as
section metadata, quoted replies stripped, one message a section -- is the next piece
and needs a real mailbox to test on.

**The room in the nebula.** "When I load a cartridge, dim the parts of the nebula that
don't pertain to it and highlight the ones that do." Nodes now carry their cartridge
ids (`k`) from `/api/library/web`; seating sets `web.room`, the dimming eases in over a
second, volumes outside the room keep 12% of their light and their edges follow the
dimmer endpoint, volumes inside gain half again, and the camera settles on the room's
centroid. Lifting the cartridge eases it back. A cartridge seated when the page opens
is the room from the first frame. On RTS the effect is a plum lattice of 391 volumes
standing in front of the HackTricks core as a ghost -- which is also the first honest
picture of how the two collections interleave.

**The nebula at 1,600.** "It's so laggy, why?" Two loops written for 176 volumes: the
force tick compared every pair (1.3 million tests a frame), and the draw made a fresh
radial gradient per node and a separate stroke per edge, sixty times a second. Now:
repulsion through a 220-unit grid (only neighbouring cells can feel each other); edges
sorted into twelve alpha steps and stroked as twelve paths, lit edges as one more; the
haze as a 64-px disc rendered once per colour and scaled into place; and the idle turn
at 30 fps rather than 60 once the simulation has settled. Measured in the tab with the
simulation still running: tick 13.7 ms, draw 2.3 ms, 56 fps on 1,611 volumes and
10,778 edges. It looks the same.

**The nebula on the GPU.** "Smoother, without losing the visual fidelity" -- and "I want
it to look the same way." The picture is unchanged and is now drawn by WebGL
(`web/nebula-gl.js`, three.js already being here for the cartridges): the page still
projects every volume and works out each one's colour and alpha with the formulas it
always used, and writes them into typed arrays; the module draws them in three passes --
threads as line geometry in the same twelve alpha steps the 2D drawing stroked (a
stencil keeps crossings within a step from darkening each other, as one stroked path
did), haze as a cloud of soft points, volumes as a cloud of hard points -- and the 2D
canvas above it keeps only the annotated rings, the finding under the eye, and the
retrieval scan. Matching it took pixel readings rather than eyeballing: the canvas
gradient interpolated colour and alpha separately, so its night disc fell off as the
square and its day disc linearly, and the 2D `lighter` accumulated coverage linearly;
the shader does both. Pixels at the core now agree to within anti-aliasing. And the
simulation no longer restarts on a whim: settled positions are kept in the browser and
the cloud opens where it was left; newcomers get a half-strength settle, a rebuilt thread
set a nudge, and a full shake only when most of the cloud is new. Without WebGL the 2D
drawing stands as it was.
