# Troubleshooting

What has gone wrong before, what it looked like, and what to do. The Settings tab's
*troubleshoot* button reads this file (and the README, DEVLOG and scripts) to advise on
an incident; keep entries concrete: symptom, cause, commands.

## First checks

Everything the library needs, in one line:

```sh
./library status          # is the API up, is the worker running
pg_isready                # Postgres
redis-cli ping            # Redis → PONG
curl -sf localhost:11434/api/tags >/dev/null && echo ollama ok   # Ollama through the tunnel
```

Then the health endpoint, which also counts orphaned vectors and stray files:

```sh
curl -s localhost:8077/api/health | python3 -m json.tool
```

Logs are in `ops/logs/api.log` and `ops/logs/worker.log`. `./library restart` restarts
both API and worker; the worker must be restarted after any code change because it
imports the task functions at start.

## Ollama is not reachable / connection refused on 11434

Symptom: `httpx.ConnectError`, `Connection refused`, chat says "the library is not
answering", reading jobs sit at *running* with no progress, `/api/health` reports
`ollama: false`.

Cause: the SSH tunnel to the GPU box has dropped, or was never opened. `localhost:11434`
is the *remote* Ollama forwarded over SSH; there is no local Ollama to start.

Do: re-open the tunnel in its own terminal, then confirm with
`curl -sf localhost:11434/api/tags`. Reading jobs resume on their own; a chat turn that
failed must be asked again. Do not start, stop or kill a local Ollama.

## Ollama is reachable but a call hangs or times out

Symptom: `ReadTimeout` after 300 s on a structured call, or a chat that never produces a
token; other calls queue behind it.

Cause: Ollama occasionally wedges on a single grammar-constrained generation. The client
aborts the request at 300 s (900 s for chat) and Ollama cancels the generation when the
client disconnects, so the next call gets through. A hung structured call is a lost item
(one section summary, one contradiction check), not a lost hour; the pass is re-runnable.

Do: nothing for a single timeout. If every call times out, check the remote box's load
(`curl -s localhost:11434/api/ps` shows what is resident) and let the owner of the box
restart Ollama *there*.

## Model returned placeholder values / unusable output

Symptom: `OllamaError: model ... returned placeholder values` or `returned unusable
output: Expecting ',' delimiter`.

Cause: two different things. Placeholders (`"..."`, `"string"`) mean the context window
was too small for the prompt and the model emitted schema-shaped filler; the client
doubles `num_ctx` and retries twice before giving up. Truncated JSON means the output hit
the `num_predict` cap — usually a runaway generation on an unbounded array.

Do: a single failure is retried automatically and the item is skipped if it keeps
failing; re-run the pass (*have it read*, *rebuild threads*, *reshelve*). If one document
fails every time, open it in the reader and check the extraction — a wordlist or a
scanned PDF with no text layer produces prompts the model cannot answer.

## "does not support thinking" (HTTP 400 from Ollama)

Symptom: chat on the *technical* model fails with a 400 mentioning thinking.

Cause: `think=true` sent to a model that lacks the capability. The client checks
`/api/show` capabilities per model and caches the answer; a stale cache after a model was
re-pulled can trip this.

Do: `./library restart` clears the cache. If it persists, `curl -s localhost:11434/api/show
-d '{"model":"<name>"}'` and look at `capabilities`.

## Postgres is not running / connection refused on 5432

Symptom: `psycopg.OperationalError: connection refused`, the API fails to start,
`./library` prints `✗ Postgres is not running`.

Do: `brew services start postgresql@16` (or however Postgres is run here), then
`pg_isready`. If the database is missing: `createdb library_agent` then
`uv run alembic upgrade head`.

## Redis is not running

Symptom: `redis.exceptions.ConnectionError`, jobs never start, `./library` prints
`✗ Redis is not running`.

Do: `brew services start redis`, then `redis-cli ping` → `PONG`. Queued jobs are stored
in Redis; if it was wiped, re-queue with `curl -X POST localhost:8077/api/read/backfill?tier=1`.

## Worker not running / jobs stay queued

Symptom: `./library status` says `worker not running`; the *In hand* panel shows jobs
waiting forever.

Do: `./library restart`. The worker log is `ops/logs/worker.log`. The worker runs one
job at a time on purpose (the models are one shared resource), so a long backfill *will*
look slow: minutes per document.

## Migration failed / column does not exist

Symptom: `sqlalchemy.exc.ProgrammingError: column ... does not exist` or
`relation ... does not exist` after pulling new code.

Cause: the database schema is behind the code.

Do: `uv run alembic upgrade head`, then `./library restart`. `uv run alembic current`
shows where the database is; `uv run alembic heads` where it should be.

## Orphaned vectors or stray files in health

Symptom: `/api/health` shows non-zero `orphan_*_vectors` or `stray_files`; the masthead
shows *N orphaned* in amber.

Cause: a document removed by something other than the API (a crashed ingest, a manual
SQL delete). `delete_document` in `db/purge.py` is the only correct way to remove one.

Do: `curl -X POST localhost:8077/api/settings/gc` removes orphaned vectors and stray
files (also the *collect garbage* button under Settings › Maintenance). Safe any time.

## Endpoint protection flagged a file in the store

Symptom: the security agent quarantines a file under `~/.library-agent/documents/`.

Cause: the store is content-addressed and keeps the original bytes of every document;
a payload collection can contain test strings (EICAR) that are harmless but signature-
matched. Files under 200 bytes are refused at import for this reason; files that were
already stored before that guard may remain.

Do: remove the document from the shelf (its stored file goes with it), or delete the
file — nothing references a stored file except its own document row.

## A cartridge will not insert

Symptom: `not a cartridge`, `content hash does not match`, `format N is newer than this
library understands`, `original with unexpected type`.

Cause, in order: the zip has no `cartridge.json`; the zip was altered or damaged after
export; it was made by a newer version of the library; a full-level cartridge carries a
file type the library does not read.

Do: ask for a fresh export; update this library if the format is newer. There is no
override — an altered cartridge is refused by design.

## The shelf is a mess / volumes on the wrong shelf

Symptom: too many sub-shelves, near-duplicate names, volumes filed oddly.

Cause: the shelving pass is a model judgement and varies run to run.

Do: press *reshelve* in the shelf header (two-step, rebuilds the whole shelf, a few
minutes), or *shelve now* under *Not yet shelved* to place only the unplaced.

## Chat answers without drawing on the shelf

Symptom: the desk says *answered without drawing on the shelf*; no margin notes.

Cause: retrieval found nothing it judged relevant, or the scope (subject chips, a
cartridge room) excluded everything relevant.

Do: clear the chips and the rack scope and ask again; try *Find* with the exact term
to see whether the passage exists at all.

## When to open an issue

Open one when the steps above do not explain it, or when a traceback points into the
library's own code (`src/library_agent/...`) rather than a service being down. Include
the incident's kind and message, the traceback tail, what you tried, and the model
names in use. Leave out paths under your home directory and anything key-shaped.
