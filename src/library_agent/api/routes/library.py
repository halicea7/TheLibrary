"""Library-level endpoints: cross-document themes, citation graph, contradictions."""

from __future__ import annotations

import uuid

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from library_agent.config import settings
from library_agent.db.session import SessionDep
from library_agent.library.citations import hubs
from library_agent.library.contradictions import list_contradictions

router = APIRouter(prefix="/api/library", tags=["library"])


class ClusterOut(BaseModel):
    id: uuid.UUID
    label: str | None
    summary: str | None
    size: int
    document_count: int
    has_contradiction: bool
    documents: list[str]


class GraphEdge(BaseModel):
    citing: str
    cited: str
    confidence: float


@router.get("/clusters", response_model=list[ClusterOut])
async def clusters(db: SessionDep, cross_document_only: bool = True) -> list[ClusterOut]:
    rows = (
        await db.execute(
            text("""
            select c.id, c.label, a.text as summary, c.size, c.document_count,
                   c.has_contradiction,
                   array_agg(distinct d.title) as docs
            from cluster c
            left join artifact a on a.target_id = c.id and a.kind = 'cluster_summary'
            join cluster_member m on m.cluster_id = c.id
            join document d on d.id = m.document_id
            where (not :cross_only or c.document_count > 1)
              and a.text is not null
            group by c.id, c.label, a.text, c.size, c.document_count, c.has_contradiction
            order by c.has_contradiction desc, c.document_count desc, c.size desc
            """),
            {"cross_only": cross_document_only},
        )
    ).all()
    return [
        ClusterOut(
            id=r.id,
            label=r.label,
            summary=r.summary,
            size=r.size,
            document_count=r.document_count,
            has_contradiction=r.has_contradiction,
            documents=list(r.docs),
        )
        for r in rows
    ]


@router.get("/contradictions")
async def contradictions(db: SessionDep) -> list[dict]:
    return await list_contradictions(db)


@router.get("/graph")
async def graph(db: SessionDep) -> dict[str, object]:
    rows = (
        await db.execute(
            text("""
            select src.title as citing, dst.title as cited, c.confidence
            from citation c
            join document src on src.id = c.citing_document_id
            join document dst on dst.id = c.matched_document_id
            order by c.confidence desc
            """)
        )
    ).all()
    return {
        "edges": [
            GraphEdge(citing=r.citing, cited=r.cited, confidence=float(r.confidence)).model_dump()
            for r in rows
        ],
        "hubs": [{"title": t, "cited_by": n} for t, n in await hubs(db)],
    }


@router.post("/rebuild")
async def rebuild(kinds: str = "all") -> dict[str, str]:
    """Queue the library-layer passes now. Reads also schedule this automatically,
    debounced; a manual rebuild supersedes any pending automatic one."""
    from library_agent.worker.tasks import REBUILD_KEY

    redis = await create_pool(RedisSettings.from_dsn(settings().redis_url))
    try:
        await redis.delete(REBUILD_KEY)
        if kinds == "all":
            await redis.enqueue_job("build_library_layer", "all")
        else:
            for kind in [k.strip() for k in kinds.split(",") if k.strip()]:
                await redis.enqueue_job("build_library_layer", kind)
    finally:
        await redis.aclose()
    return {"queued": kinds}
