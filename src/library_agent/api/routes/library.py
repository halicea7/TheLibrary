"""Library-level endpoints: cross-document themes, citation graph, contradictions."""

from __future__ import annotations

import uuid

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from library_agent.config import settings
from library_agent.db.session import SessionDep
from library_agent.library.citations import hubs
from library_agent.library.contradictions import list_contradictions
from library_agent.retrieval.threads import scope_documents, scope_thread, scoped_claims

router = APIRouter(prefix="/api/library", tags=["library"])


class ClusterOut(BaseModel):
    id: uuid.UUID
    label: str | None
    summary: str | None
    size: int
    document_count: int
    has_contradiction: bool
    documents: list[str]
    document_ids: list[str] = []
    # Inside a scope: how many of its volumes are in it, and what those volumes claim. The
    # summary is the whole library's; these are what the scope holds.
    scoped: dict | None = None
    claims: list[dict] = []


class GraphEdge(BaseModel):
    citing: str
    cited: str
    confidence: float


async def _scope_docs(
    db: SessionDep, categories: str | None, cartridges: str | None
) -> set[uuid.UUID] | None:
    """The volumes the chips and the rack admit, by the rule Write uses too."""
    return await scope_documents(
        db,
        [uuid.UUID(x) for x in categories.split(",") if x.strip()] if categories else None,
        [uuid.UUID(x) for x in cartridges.split(",") if x.strip()] if cartridges else None,
    )


@router.get("/clusters", response_model=list[ClusterOut])
async def clusters(
    db: SessionDep,
    cross_document_only: bool = True,
    categories: str | None = None,
    cartridges: str | None = None,
) -> list[ClusterOut]:
    docs = await _scope_docs(db, categories, cartridges)
    rows = (
        await db.execute(
            text("""
            select c.id, c.label, a.text as summary, c.size, c.document_count,
                   c.has_contradiction, a.data->'claims' as listed,
                   a.data->'claim_sources' as listed_sources,
                   array_agg(distinct d.title) as docs,
                   array_agg(distinct d.id) as doc_ids
            from cluster c
            left join artifact a on a.target_id = c.id and a.kind = 'cluster_summary'
            join cluster_member m on m.cluster_id = c.id
            join document d on d.id = m.document_id
            where (not :cross_only or c.document_count > 1)
              and a.text is not null
            group by c.id, c.label, a.text, a.data, c.size, c.document_count,
                     c.has_contradiction
            order by c.document_count desc, c.size desc
            """),
            {"cross_only": cross_document_only},
        )
    ).all()
    if docs is not None:
        # Rebuilt from inside the scope: its in-scope claims and volumes, or dropped when
        # fewer than two of its volumes are in scope.
        claims = await scoped_claims(db, [r.id for r in rows if set(r.doc_ids) & docs], docs)
        out = []
        for r in rows:
            if not set(r.doc_ids) & docs:
                continue
            t = scope_thread(
                {
                    "documents": list(r.docs),
                    "_doc_ids": list(r.doc_ids),
                    "_listed": list(r.listed or []),
                    "_listed_sources": list(r.listed_sources or []),
                    "document_count": r.document_count,
                    "contradiction": None,
                },
                claims.get(r.id),
                docs,
            )
            if not t:
                continue
            in_ids = [str(x) for x in r.doc_ids if x in docs]
            out.append(
                ClusterOut(
                    id=r.id,
                    label=r.label,
                    summary=r.summary,
                    size=r.size,
                    document_count=r.document_count,
                    has_contradiction=r.has_contradiction,
                    documents=t["documents"],
                    document_ids=in_ids,
                    scoped=t["scoped"],
                    claims=t["claims"],
                )
            )
        return out
    return [
        ClusterOut(
            id=r.id,
            label=r.label,
            summary=r.summary,
            size=r.size,
            document_count=r.document_count,
            has_contradiction=r.has_contradiction,
            documents=list(r.docs),
            document_ids=[str(x) for x in r.doc_ids],
        )
        for r in rows
    ]


@router.get("/contradictions")
async def contradictions(
    db: SessionDep, categories: str | None = None, cartridges: str | None = None
) -> list[dict]:
    out = await list_contradictions(db)
    docs = await _scope_docs(db, categories, cartridges)
    if docs is not None:
        # A disagreement is in scope when both of its sides are.
        titles = set(
            (
                await db.execute(
                    text("select title from document where id = any(:d)"), {"d": list(docs)}
                )
            ).scalars()
        )
        out = [
            x for x in out if x.get("pair") and all(side["source"] in titles for side in x["pair"])
        ]
    return out


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


_WEB_NODES = """
select d.id, d.title, d.tier, d.kind,
       (select count(*) from chunk c where c.document_id = d.id) as chunks,
       top.name as top_shelf, sub.name as sub_shelf,
       cart.colour as cartridge_colour,
       (select array_agg(cd2.cartridge_id::text) from cartridge_document cd2
         where cd2.document_id = d.id) as cartridge_ids
from document d
left join category sub on sub.id = d.shelf_id
left join category top on top.id = sub.parent_id
left join lateral (
    select ca.colour from cartridge_document cd join cartridge ca on ca.id = cd.cartridge_id
    where cd.document_id = d.id order by ca.imported_at limit 1
) cart on true
where d.status = 'ready'
"""

# Nearest neighbours by the document vector (the Tier 1 summary once read, the Tier 0
# fingerprint before). Three per volume: enough to knit the web, few enough to read.
_WEB_NEAR = """
with vec as (
    select coalesce(a.target_id, e.owner_id) as document_id, e.vec
    from embedding e
    left join artifact a on a.id = e.owner_id and a.kind = 'document_summary'
    where e.model = :m
      and ((e.owner_kind = 'artifact' and a.id is not null) or e.owner_kind = 'document')
),
best as (
    select distinct on (document_id) document_id, vec from vec
    order by document_id, (select 1)  -- either vector will do; prefer whichever sorts first
)
select v.document_id as a, n.document_id as b, 1 - (v.vec <=> n.vec) as sim
from best v
join lateral (
    select w.document_id, w.vec from best w
    where w.document_id <> v.document_id
    order by w.vec <=> v.vec limit 3
) n on true
where 1 - (v.vec <=> n.vec) > 0.55
"""

_WEB_THREADS = """
select a.document_id as a, b.document_id as b, count(*) as shared
from cluster_member a join cluster_member b
  on a.cluster_id = b.cluster_id and a.document_id < b.document_id
group by 1, 2
"""

_WEB_CITES = """
select citing_document_id as a, matched_document_id as b
from citation where matched_document_id is not null
"""


@router.get("/web")
async def web(db: SessionDep) -> dict[str, object]:
    """The library as a web: one node per volume, edges where volumes cite each other,
    share a thread, or simply sit close in meaning. It is what the Ask pane is drawn over,
    and what lights up when passages are retrieved."""
    nodes = [
        {
            "id": str(r.id),
            "t": r.title,
            "tier": r.tier,
            "n": r.chunks,
            "top": r.top_shelf,
            "sub": r.sub_shelf,
            "c": r.cartridge_colour,
            "k": r.cartridge_ids or [],  # the rooms this volume belongs to
        }
        for r in await db.execute(text(_WEB_NODES))
    ]
    edges: dict[tuple[str, str], dict] = {}

    def add(a, b, kind, w):
        key = (min(str(a), str(b)), max(str(a), str(b)))
        e = edges.setdefault(key, {"a": key[0], "b": key[1], "k": kind, "w": 0.0})
        e["w"] = max(e["w"], w)
        if (
            kind == "cite" or kind == "thread" and e["k"] != "cite"
        ):  # the strongest kind wins the label
            e["k"] = kind

    for r in await db.execute(text(_WEB_CITES)):
        add(r.a, r.b, "cite", 1.0)
    for r in await db.execute(text(_WEB_THREADS)):
        add(r.a, r.b, "thread", min(1.0, 0.5 + 0.15 * r.shared))
    for r in await db.execute(text(_WEB_NEAR), {"m": settings().embed_model}):
        add(r.a, r.b, "near", float(r.sim))
    return {"nodes": nodes, "edges": list(edges.values())}


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


def same_source(title: str, source: str) -> bool:
    """A disagreement names its sides as the judge saw them -- "Title — what it is about",
    the title sometimes cut short -- so match on the title part, allowing a clipped one."""
    name = source.split(" — ")[0].strip().lower()
    t = (title or "").strip().lower()
    if not name or not t:
        return False
    return t == name or (len(name) >= 12 and (t.startswith(name) or name.startswith(t)))


@router.get("/evidence")
async def evidence(db: SessionDep, cluster: uuid.UUID, source: str, claim: str) -> dict:
    """The passage behind one claim of a thread, opened from what the library stored rather
    than searched for again: the claim's own record in the thread, the section it was drawn
    from, and within that section the passage nearest the claim's kept vector. `exact` is
    false when the claim was matched loosely or no vector was kept for it."""
    from library_agent.library.cluster import _hash

    want = " ".join(claim.split())
    rows = (
        await db.execute(
            text(
                "select cc.artifact_id, cc.text, cc.claim_hash, a.target_id as section_id,"
                " d.title from cluster_claim cc join document d on d.id = cc.document_id"
                " join artifact a on a.id = cc.artifact_id where cc.cluster_id = :c"
            ),
            {"c": cluster},
        )
    ).all()
    if not rows:  # a thread built before claims were kept: the member sections' own lists
        rows = (
            await db.execute(
                text(
                    "select a.id as artifact_id, claim.value as text, null as claim_hash,"
                    " a.target_id as section_id, d.title"
                    " from cluster_member m join document d on d.id = m.document_id"
                    " join artifact a on a.id = m.artifact_id,"
                    " lateral jsonb_array_elements_text(a.data->'claims') as claim"
                    " where m.cluster_id = :c"
                ),
                {"c": cluster},
            )
        ).all()
    rows = [r for r in rows if same_source(r.title, source)]
    if not rows:
        raise HTTPException(404, "that claim is not in this thread")

    def closeness(t: str) -> float:
        a, b = set(t.lower().split()), set(want.lower().split())
        return 2.0 if " ".join(t.split()) == want else len(a & b) / max(1, len(a | b))

    best = max(rows, key=lambda r: closeness(r.text))
    exact = closeness(best.text) == 2.0
    h = best.claim_hash or _hash(best.text)
    hit = (
        await db.execute(
            text(
                "select c.id, c.document_id, d.title, c.page_start, s.path"
                " from chunk c join document d on d.id = c.document_id"
                " left join section s on s.id = c.section_id"
                " join embedding e on e.owner_kind = 'chunk' and e.owner_id = c.id"
                " join claim_vector v on v.hash = :h and v.model = e.model"
                " where c.section_id = :s and c.kind = 'text'"
                " order by e.vec <=> v.vec limit 1"
            ),
            {"h": h, "s": best.section_id},
        )
    ).first()
    if hit is None:  # no kept vector for the claim: the section's opening passage
        exact = False
        hit = (
            await db.execute(
                text(
                    "select c.id, c.document_id, d.title, c.page_start, s.path"
                    " from chunk c join document d on d.id = c.document_id"
                    " left join section s on s.id = c.section_id"
                    " where c.section_id = :s and c.kind = 'text' order by c.order_index limit 1"
                ),
                {"s": best.section_id},
            )
        ).first()
    if hit is None:
        raise HTTPException(404, "the section behind this claim has no passages")
    return {
        "chunk_id": str(hit.id),
        "document_id": str(hit.document_id),
        "document_title": hit.title,
        "page": hit.page_start,
        "section": hit.path,
        "claim": best.text,
        "exact": exact,
    }
