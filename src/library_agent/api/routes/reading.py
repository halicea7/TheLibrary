"""Reading jobs and categories."""

from __future__ import annotations

import uuid

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from library_agent.config import settings
from library_agent.db.models import Category, Document, DocumentCategory, Job
from library_agent.db.session import SessionDep
from library_agent.worker.tasks import enqueue_read, enqueue_reshelve

router = APIRouter(prefix="/api", tags=["reading"])


class JobOut(BaseModel):
    id: uuid.UUID
    kind: str
    document_id: uuid.UUID | None
    document_title: str | None
    state: str
    progress_current: int
    progress_total: int
    yielded_reason: str | None
    error: str | None


class CategoryOut(BaseModel):
    id: uuid.UUID
    name: str
    documents: int  # tagged with it (a top shelf counts everything beneath it)
    parent_id: uuid.UUID | None = None
    shelved: int = 0  # volumes whose one place this is


async def _redis():
    return await create_pool(RedisSettings.from_dsn(settings().redis_url))


@router.post("/documents/{document_id}/read", response_model=JobOut)
async def start_reading(document_id: uuid.UUID, db: SessionDep, tier: int = 1) -> JobOut:
    doc = (
        await db.execute(select(Document).where(Document.id == document_id))
    ).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "document not found")
    redis = await _redis()
    try:
        job_id = await enqueue_read(redis, document_id, tier=tier)
    finally:
        await redis.aclose()
    job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one()
    return JobOut(
        id=job.id,
        kind=job.kind,
        document_id=job.document_id,
        document_title=doc.title,
        state=job.state,
        progress_current=job.progress_current,
        progress_total=job.progress_total,
        yielded_reason=job.yielded_reason,
        error=job.error,
    )


@router.post("/read/backfill")
async def start_backfill(db: SessionDep, tier: int = 1) -> dict[str, int]:
    """Queue everything below `tier`. tier=2 annotates the whole shelf."""
    ids = list((await db.execute(select(Document.id).where(Document.tier < tier))).scalars())
    redis = await _redis()
    try:
        for did in ids:
            await enqueue_read(redis, did, tier=tier)
    finally:
        await redis.aclose()
    return {"queued": len(ids)}


@router.get("/jobs", response_model=list[JobOut])
async def list_jobs(db: SessionDep, limit: int = 25) -> list[JobOut]:
    rows = (
        await db.execute(
            select(Job, Document.title)
            .outerjoin(Document, Document.id == Job.document_id)
            .order_by(Job.created_at.desc())
            .limit(limit)
        )
    ).all()
    return [
        JobOut(
            id=j.id,
            kind=j.kind,
            document_id=j.document_id,
            document_title=title,
            state=j.state,
            progress_current=j.progress_current,
            progress_total=j.progress_total,
            yielded_reason=j.yielded_reason,
            error=j.error,
        )
        for j, title in rows
    ]


@router.get("/categories", response_model=list[CategoryOut])
async def list_categories(db: SessionDep) -> list[CategoryOut]:
    rows = (
        await db.execute(
            select(Category, func.count(DocumentCategory.document_id))
            .outerjoin(DocumentCategory, DocumentCategory.category_id == Category.id)
            .where(Category.canonical.is_(True))
            .group_by(Category.id)
            .order_by(func.count(DocumentCategory.document_id).desc(), Category.name)
        )
    ).all()
    shelved = dict(
        (
            await db.execute(
                select(Document.shelf_id, func.count())
                .where(Document.shelf_id.is_not(None))
                .group_by(Document.shelf_id)
            )
        ).all()
    )
    out = {
        c.id: CategoryOut(
            id=c.id, name=c.name, documents=n, parent_id=c.parent_id, shelved=shelved.get(c.id, 0)
        )
        for c, n in rows
    }
    # A top shelf counts what stands beneath it, not what happens to be tagged with it.
    for c in out.values():
        if c.parent_id and c.parent_id in out:
            out[c.parent_id].shelved += c.shelved
    for c in out.values():
        if c.parent_id is None and any(x.parent_id == c.id for x in out.values()):
            c.documents = c.shelved
    return sorted(out.values(), key=lambda c: (-c.documents, c.name))


@router.post("/shelf/reshelve")
async def reshelve(db: SessionDep, rebuild: bool = True) -> dict[str, str]:
    """Organise the shelf into top shelves and sub-shelves and give every read volume
    one place. `rebuild=false` keeps the current shelves and only places the unshelved."""
    redis = await _redis()
    try:
        job_id = await enqueue_reshelve(redis, rebuild=rebuild)
    finally:
        await redis.aclose()
    return {"job_id": str(job_id)}
