"""Background reading jobs.

Tier 1 takes minutes per document, so it belongs on the queue rather than in a request.
Jobs yield to interactive chat between sections -- never mid-call, since an in-flight
generation cannot be interrupted cleanly."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, ClassVar

from arq import Retry
from arq import func as arq_func
from arq.connections import RedisSettings
from sqlalchemy import func, select, update

from library_agent.config import settings
from library_agent.db.models import Document, Job, JobState
from library_agent.db.session import session_scope
from library_agent.llm.client import LLM
from library_agent.llm.lease import PAUSED_KEY, reading_may_proceed, redis_client
from library_agent.ops.incidents import record_exception
from library_agent.reading.tier1 import run_tier1
from library_agent.reading.tier2 import run_tier2

log = logging.getLogger(__name__)

# Tier 1 on a long book legitimately runs for many minutes; ARQ's 300s default would
# kill it partway through.
JOB_TIMEOUT = 60 * 60 * 3
# A rebuild of threads on a large shelf is thousands of summaries and judgements, wholesale
# each time: at 1,600 volumes it ran past three hours, was killed, was retried from the
# start, and would have looped like that all night. Its own bound, and the reshelve's,
# is a day.
LIBRARY_TIMEOUT = 60 * 60 * 24
PAUSE_REQUEUE_SECONDS = 600  # a pause longer than this sends the job back to the queue
PAUSE_RETRY_DEFER = 300
POLL_SECONDS = 15


async def _set_job(job_id: uuid.UUID, **values: Any) -> None:
    async with session_scope() as db:
        await db.execute(update(Job).where(Job.id == job_id).values(**values))


def _make_gate(job_id: uuid.UUID):
    """Block while a chat session is active, recording why in the job row so the UI can
    say 'waiting for chat' rather than looking stalled."""

    async def gate() -> None:
        r = redis_client()
        try:
            announced = False
            waited = 0.0
            while True:
                ok, reason = await reading_may_proceed(r)
                if ok:
                    if announced:
                        await _set_job(job_id, state=JobState.RUNNING, yielded_reason=None)
                    return
                if not announced:
                    await _set_job(job_id, state=JobState.YIELDED, yielded_reason=reason)
                    announced = True
                if reason == "paused" and waited >= PAUSE_REQUEUE_SECONDS:
                    # A long pause must not sit inside the job: arq's job timeout would
                    # kill it (one did, after a nine-hour pause). Hand it back to the
                    # queue and try again later; what a read already wrote stays.
                    await _set_job(job_id, state=JobState.QUEUED, yielded_reason="paused")
                    raise Retry(defer=PAUSE_RETRY_DEFER)
                await asyncio.sleep(POLL_SECONDS)
                waited += POLL_SECONDS
        finally:
            await r.aclose()

    return gate


async def read_document(ctx: dict, document_id: str, job_id: str, tier: int = 1) -> dict[str, Any]:
    did, jid = uuid.UUID(document_id), uuid.UUID(job_id)
    async with session_scope() as db:
        if await db.get(Document, did) is None:
            # Removed from the shelf while the job waited its turn. Nothing went wrong.
            await _set_job(
                jid, state=JobState.ERROR, error="document was removed before it was read"
            )
            return {"document_id": document_id, "skipped": "removed"}
    await _set_job(jid, state=JobState.RUNNING)
    await _make_gate(jid)()  # paused, or a chat in progress: wait before starting

    async def progress(current: int, total: int) -> None:
        await _set_job(jid, progress_current=current, progress_total=total)

    try:
        async with LLM() as client, session_scope() as db:
            if tier >= 2:
                # Annotation needs the orientation card from Tier 1; run it first if absent.
                doc = await db.get(Document, did)
                if doc and doc.tier < 1:
                    await run_tier1(db, did, client=client, progress=progress, gate=_make_gate(jid))
                r2 = await run_tier2(
                    db, did, client=client, progress=progress, gate=_make_gate(jid)
                )
                await _set_job(jid, state=JobState.DONE, yielded_reason=None)
                await schedule_library_rebuild(ctx["redis"])
                return {"document_id": str(did), "reflections": r2.reflections}
            result = await run_tier1(
                db, did, client=client, progress=progress, gate=_make_gate(jid)
            )
        await _set_job(jid, state=JobState.DONE, yielded_reason=None)
        await schedule_library_rebuild(ctx["redis"])
        return {
            "document_id": str(did),
            "sections_read": result.sections_read,
            "categories": result.categories,
        }
    except Exception as exc:
        await record_exception(
            exc, source="worker", context={"job": f"tier{tier}", "document_id": str(did)}
        )
        log.exception("tier1 failed for %s", did)
        await _set_job(jid, state=JobState.ERROR, error=str(exc)[:2000])
        async with session_scope() as db:
            await db.execute(
                update(Document).where(Document.id == did).values(error=str(exc)[:2000])
            )
        raise


REBUILD_KEY = "library:rebuild_scheduled"
REBUILD_DONE_KEY = "library:rebuild_done_at"


async def _reads_pending() -> int:
    async with session_scope() as db:
        return (
            await db.execute(
                select(func.count())
                .select_from(Job)
                .where(
                    Job.kind.in_(["tier1", "tier2"]),
                    Job.state.in_([JobState.QUEUED, JobState.RUNNING, JobState.YIELDED]),
                )
            )
        ).scalar() or 0


async def build_library_layer(
    ctx: dict, kind: str, requested_at: float | None = None
) -> dict[str, Any]:
    """Cross-document passes. `all` runs the three in dependency order; each one is also
    available on its own. They hit the model, so contention with reading is real -- the
    worker runs one job at a time for exactly that reason.

    A scheduled rebuild (`requested_at` set) waits for the shelf to go quiet: if reads
    are still queued it steps back into the queue behind them rather than running a
    forty-minute pass in the middle of a folder import, and if a rebuild has already
    started since it was asked for, it is covered and does nothing. A 391-page import
    once queued seven rebuilds this way, one per debounce window, each blocking the
    reads behind it."""

    redis = ctx.get("redis")
    if requested_at is not None and redis is not None:
        # Covered if a rebuild *finished* after this one was asked for -- judged on
        # completion, not start, so a rebuild cut off by a worker restart is not
        # mistaken for done when arq retries it.
        done = await redis.get(REBUILD_DONE_KEY)
        if done and float(done) >= requested_at:
            return {"kind": kind, "skipped": "a later rebuild already ran"}
        if await _reads_pending():
            delay = settings().library_rebuild_delay_seconds
            await redis.enqueue_job(
                "build_library_layer", kind, requested_at=requested_at, _defer_by=delay
            )
            return {"kind": kind, "deferred": delay}

    kinds = ["citations", "clusters", "contradictions"] if kind == "all" else [kind]
    out: dict[str, Any] = {"kind": kind}
    if redis is not None and await redis.exists(PAUSED_KEY):
        # Paused: come back later rather than hold the job row open for hours.
        await redis.enqueue_job(
            "build_library_layer", kind, requested_at=requested_at, _defer_by=120
        )
        return {"kind": kind, "deferred": "paused"}
    # A job row, so the UI's "In hand" and anything waiting for an idle worker can see
    # that a rebuild is holding it -- these run for many minutes on a large library.
    async with session_scope() as db:
        job = Job(kind="library", state=JobState.RUNNING, progress_total=len(kinds))
        db.add(job)
        await db.flush()
        jid = job.id
    try:
        await _library_passes(jid, kinds, out)
    except asyncio.CancelledError:
        # Killed from outside (a worker restart, a timeout): say so on the row rather
        # than leave a second "running" rebuild beside the retry.
        await _set_job(jid, state=JobState.ERROR, error="cut off", yielded_reason=None)
        raise
    await _set_job(
        jid, state=JobState.DONE, progress_current=1, progress_total=1, yielded_reason=None
    )
    if redis is not None:
        await redis.set(REBUILD_DONE_KEY, str(time.time()))
    return out


async def _library_passes(jid: uuid.UUID, kinds: list[str], out: dict[str, Any]) -> None:
    from library_agent.library.citations import build_citation_graph
    from library_agent.library.cluster import build_clusters
    from library_agent.library.contradictions import find_contradictions

    async with LLM() as client:
        for n, k in enumerate(kinds):
            await _set_job(jid, progress_current=0, progress_total=1, yielded_reason=k)

            # Each pass counts its own work (documents, clusters); the row shows that
            # count under the pass name, so a twenty-minute clusters pass moves.
            async def progress(
                current: int, total: int, phase: str | None = None, k: str = k
            ) -> None:
                # A pass may name what it is on ("embedding claims") so a long warm-up
                # with nothing to count does not read as a stall.
                await _set_job(
                    jid,
                    progress_current=current,
                    progress_total=max(1, total),
                    yielded_reason=f"{k} · {phase}" if phase else k,
                )

            try:
                async with session_scope() as db:
                    if k == "citations":
                        r = await build_citation_graph(db, progress=progress)
                        out["citations"] = {"matched": r.matched, "references": r.references_found}
                    elif k == "clusters":
                        r = await build_clusters(
                            db, client=client, progress=progress, gate=_make_gate(jid)
                        )
                        out["clusters"] = {
                            "clusters": r.clusters,
                            "cross_document": r.cross_document,
                        }
                    elif k == "contradictions":
                        out["contradictions"] = await find_contradictions(
                            db, client=client, progress=progress, gate=_make_gate(jid)
                        )
            except Exception as exc:
                # One pass failing must not take the others with it.
                await record_exception(exc, source="worker", context={"job": f"library:{k}"})
                log.exception("library pass %s failed", k)
                out[k] = {"error": str(exc)[:300]}


async def reshelve_library(
    ctx: dict, job_id: str, rebuild: bool = True, category_id: str | None = None
) -> dict[str, Any]:
    """Organise the shelf into two levels and put every read volume in one place."""
    from library_agent.library.shelving import reshelve

    jid = uuid.UUID(job_id)
    await _set_job(jid, state=JobState.RUNNING)

    async def progress(current: int, total: int, phase: str | None = None) -> None:
        await _set_job(
            jid, progress_current=current, progress_total=total, yielded_reason=phase or None
        )

    try:
        async with LLM() as client, session_scope() as db:
            r = await reshelve(
                db,
                client=client,
                rebuild=rebuild,
                progress=progress,
                gate=_make_gate(jid),
                category_id=uuid.UUID(category_id) if category_id else None,
            )
        await _set_job(jid, state=JobState.DONE, yielded_reason=None)
        return r.__dict__
    except Exception as exc:
        await record_exception(exc, source="worker", context={"job": "reshelve"})
        log.exception("reshelve failed")
        await _set_job(jid, state=JobState.ERROR, error=str(exc)[:2000])
        raise


async def enqueue_reshelve(
    redis, *, rebuild: bool = True, category_id: uuid.UUID | None = None
) -> uuid.UUID:
    async with session_scope() as db:
        job = Job(kind="reshelve", state=JobState.QUEUED)
        db.add(job)
        await db.flush()
        job_id = job.id
    await redis.enqueue_job(
        "reshelve_library",
        str(job_id),
        rebuild=rebuild,
        category_id=str(category_id) if category_id else None,
    )
    return job_id


async def schedule_library_rebuild(redis) -> bool:
    """Debounced: the first read to finish in a quiet window schedules one rebuild for
    `library_rebuild_delay_seconds` later; reads that finish inside the window do nothing.
    Returns whether this call scheduled it."""
    delay = settings().library_rebuild_delay_seconds
    # SET NX with a TTL is the whole debounce: the key is the reservation.
    if not await redis.set(REBUILD_KEY, "1", ex=delay, nx=True):
        return False
    await redis.enqueue_job("build_library_layer", "all", requested_at=time.time(), _defer_by=delay)
    return True


async def describe_figures_job(ctx: dict, document_id: str, job_id: str) -> dict[str, Any]:
    """The vision pass alone, for PDFs read before it existed."""
    from library_agent.reading.figures import describe_figures

    did, jid = uuid.UUID(document_id), uuid.UUID(job_id)
    await _set_job(jid, state=JobState.RUNNING)
    try:
        async with LLM() as client, session_scope() as db:
            r = await describe_figures(db, did, client=client, gate=_make_gate(jid))
        await _set_job(jid, state=JobState.DONE, yielded_reason=None)
        return {"document_id": document_id, "figures": r.figures, "described": r.described}
    except Exception as exc:
        await record_exception(
            exc, source="worker", context={"job": "figures", "document": document_id}
        )
        log.exception("figure pass failed")
        await _set_job(jid, state=JobState.ERROR, error=str(exc)[:2000])
        raise


async def enqueue_figures(redis, document_id: uuid.UUID) -> uuid.UUID:
    async with session_scope() as db:
        job = Job(kind="figures", document_id=document_id, state=JobState.QUEUED)
        db.add(job)
        await db.flush()
        job_id = job.id
    await redis.enqueue_job("describe_figures_job", str(document_id), str(job_id))
    return job_id


async def enqueue_read(redis, document_id: uuid.UUID, *, tier: int = 1) -> uuid.UUID:
    """Create the Job row first so the UI has something to poll immediately."""
    async with session_scope() as db:
        job = Job(kind=f"tier{tier}", document_id=document_id, tier=tier, state=JobState.QUEUED)
        db.add(job)
        await db.flush()
        job_id = job.id
    await redis.enqueue_job("read_document", str(document_id), str(job_id), tier=tier)
    return job_id


async def backfill(ctx: dict, tier: int = 1) -> dict[str, Any]:
    """Queue every document below `tier` for reading. This is the overnight path."""
    async with session_scope() as db:
        ids = list((await db.execute(select(Document.id).where(Document.tier < tier))).scalars())
    for did in ids:
        await enqueue_read(ctx["redis"], did, tier=tier)
    return {"queued": len(ids)}


async def _startup(ctx: dict) -> None:
    from library_agent.ops.incidents import install_handler

    install_handler("worker")


class WorkerSettings:
    functions: ClassVar[list] = [
        read_document,
        backfill,
        arq_func(build_library_layer, timeout=LIBRARY_TIMEOUT),
        arq_func(reshelve_library, timeout=LIBRARY_TIMEOUT),
        describe_figures_job,
    ]
    on_startup = _startup
    redis_settings = RedisSettings.from_dsn(settings().redis_url)
    job_timeout = JOB_TIMEOUT
    # Retry is how a paused job returns to the queue every few minutes; a pause of a
    # night is many retries. Genuine failures are never retried by arq, so this only
    # bounds how long a pause can last.
    max_tries = 10_000
    # One at a time: the models are a single shared resource, so concurrency here would
    # only make every job slower and starve chat.
    max_jobs = 1
    keep_result = 3600
