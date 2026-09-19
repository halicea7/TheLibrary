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
