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

from arq.connections import RedisSettings
from sqlalchemy import func, select, update

from library_agent.config import settings
from library_agent.db.models import Document, Job, JobState
from library_agent.db.session import session_scope
from library_agent.llm.lease import reading_may_proceed, redis_client
from library_agent.llm.ollama import Ollama
from library_agent.ops.incidents import record_exception
from library_agent.reading.tier1 import run_tier1
from library_agent.reading.tier2 import run_tier2

log = logging.getLogger(__name__)

# Tier 1 on a long book legitimately runs for many minutes; ARQ's 300s default would
# kill it partway through.
JOB_TIMEOUT = 60 * 60 * 3
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
            while True:
                ok, reason = await reading_may_proceed(r)
                if ok:
                    if announced:
                        await _set_job(job_id, state=JobState.RUNNING, yielded_reason=None)
                    return
                if not announced:
                    await _set_job(job_id, state=JobState.YIELDED, yielded_reason=reason)
                    announced = True
                await asyncio.sleep(POLL_SECONDS)
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

    async def progress(current: int, total: int) -> None:
        await _set_job(jid, progress_current=current, progress_total=total)

    try:
        async with Ollama() as client, session_scope() as db:
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
REBUILD_STARTED_KEY = "library:rebuild_started_at"


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
    from library_agent.library.citations import build_citation_graph
    from library_agent.library.cluster import build_clusters
    from library_agent.library.contradictions import find_contradictions

    redis = ctx.get("redis")
    if requested_at is not None and redis is not None:
        started = await redis.get(REBUILD_STARTED_KEY)
        if started and float(started) >= requested_at:
            return {"kind": kind, "skipped": "a later rebuild already ran"}
        if await _reads_pending():
            delay = settings().library_rebuild_delay_seconds
            await redis.enqueue_job(
                "build_library_layer", kind, requested_at=requested_at, _defer_by=delay
            )
            return {"kind": kind, "deferred": delay}
        await redis.set(REBUILD_STARTED_KEY, str(time.time()))

    kinds = ["citations", "clusters", "contradictions"] if kind == "all" else [kind]
    out: dict[str, Any] = {"kind": kind}
    # A job row, so the UI's "In hand" and anything waiting for an idle worker can see
    # that a rebuild is holding it -- these run for many minutes on a large library.
    async with session_scope() as db:
        job = Job(kind="library", state=JobState.RUNNING, progress_total=len(kinds))
        db.add(job)
        await db.flush()
        jid = job.id
    async with Ollama() as client:
        for n, k in enumerate(kinds):
            await _set_job(jid, progress_current=0, progress_total=1, yielded_reason=k)

            # Each pass counts its own work (documents, clusters); the row shows that
            # count under the pass name, so a twenty-minute clusters pass moves.
            async def progress(current: int, total: int) -> None:
                await _set_job(jid, progress_current=current, progress_total=max(1, total))

            try:
                async with session_scope() as db:
                    if k == "citations":
                        r = await build_citation_graph(db, progress=progress)
                        out["citations"] = {"matched": r.matched, "references": r.references_found}
                    elif k == "clusters":
                        r = await build_clusters(db, client=client, progress=progress)
                        out["clusters"] = {
                            "clusters": r.clusters,
                            "cross_document": r.cross_document,
                        }
                    elif k == "contradictions":
                        out["contradictions"] = await find_contradictions(
                            db, client=client, progress=progress
                        )
            except Exception as exc:
                # One pass failing must not take the others with it.
                await record_exception(exc, source="worker", context={"job": f"library:{k}"})
                log.exception("library pass %s failed", k)
                out[k] = {"error": str(exc)[:300]}
    await _set_job(
        jid, state=JobState.DONE, progress_current=1, progress_total=1, yielded_reason=None
    )
    return out


async def reshelve_library(
    ctx: dict, job_id: str, rebuild: bool = True, category_id: str | None = None
) -> dict[str, Any]:
    """Organise the shelf into two levels and put every read volume in one place."""
    from library_agent.library.shelving import reshelve

    jid = uuid.UUID(job_id)
    await _set_job(jid, state=JobState.RUNNING)

    async def progress(current: int, total: int) -> None:
        await _set_job(jid, progress_current=current, progress_total=total)

    try:
        async with Ollama() as client, session_scope() as db:
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
    functions: ClassVar[list] = [read_document, backfill, build_library_layer, reshelve_library]
    on_startup = _startup
    redis_settings = RedisSettings.from_dsn(settings().redis_url)
    job_timeout = JOB_TIMEOUT
    # One at a time: the models are a single shared resource, so concurrency here would
    # only make every job slower and starve chat.
    max_jobs = 1
    keep_result = 3600
