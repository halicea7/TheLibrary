"""The library's cross-document structure, as material for planning a long-form document.

Threads (clusters that span more than one volume) and the disagreements among them are
what a literature review is made of: here is what several works say about one thing, and
here is where they part. The compose planner is handed the threads nearest a brief so the
document is built around the connections the library already found, not re-derived one
passage at a time. Contradictions ride along on the threads that carry them."""

from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.llm.embed import embed_query

_SQL = """
select cl.id, cl.label, a.text as summary, cl.document_count, cl.has_contradiction,
       a.data->'claims' as listed, a.data->'claim_sources' as listed_sources,
       (select array_agg(distinct d.title)
        from cluster_member m join document d on d.id = m.document_id
        where m.cluster_id = cl.id) as documents,
       (select array_agg(distinct m.document_id)
        from cluster_member m where m.cluster_id = cl.id) as document_ids
from embedding e
join artifact a on a.id = e.owner_id and a.kind = 'cluster_summary'
join cluster cl on cl.id = a.target_id
where e.owner_kind = 'artifact' and e.model = :emodel
  and cl.document_count > 1
  and (cast(:docs as uuid[]) is null or exists (
      select 1 from cluster_member m
      where m.cluster_id = cl.id and m.document_id = any(cast(:docs as uuid[]))
  ))
order by e.vec <=> cast(:qv as halfvec)
limit :lim
"""

# How many in-scope claims a scoped thread shows in place of its whole-library summary.
SCOPED_CLAIMS = 8


async def scope_documents(
    db: AsyncSession,
    category_ids: list[uuid.UUID] | None = None,
    cartridge_ids: list[uuid.UUID] | None = None,
    levels: list[str] | None = None,
) -> set[uuid.UUID] | None:
    """The volumes a scope admits, by one rule for every view of the threads: a shelf
    stands for its sub-shelves, and a volume is on it by home shelf or by subject; a
    cartridge admits the volumes it carries; both together admit what both admit; a
    classification ceiling (`levels`) admits the volumes at or below it. None when nothing
    is scoped."""
    if not category_ids and not cartridge_ids and not levels:
        return None
    from library_agent.library.shelving import expand_category_ids

    keep: set[uuid.UUID] | None = None
    if category_ids:
        cats = await expand_category_ids(db, list(category_ids))
        keep = set(
            (
                await db.execute(
                    text(
                        "select d.id from document d where d.shelf_id = any(:c) or exists ("
                        "select 1 from document_category dc where dc.document_id = d.id"
                        " and dc.category_id = any(:c))"
                    ),
                    {"c": cats},
                )
            ).scalars()
        )
    if cartridge_ids:
        carts = set(
            (
                await db.execute(
                    text("select document_id from cartridge_document where cartridge_id = any(:c)"),
                    {"c": list(cartridge_ids)},
                )
            ).scalars()
        )
        keep = carts if keep is None else keep & carts
    if levels:
        from library_agent.classification import default_level

        cleared = set(
            (
                await db.execute(
                    text(
                        "select id from document"
                        " where coalesce(classification, cast(:d as text)) = any(cast(:l as text[]))"
                    ),
                    {"d": default_level(), "l": list(levels)},
                )
            ).scalars()
        )
        keep = cleared if keep is None else keep & cleared
    return keep or set()


async def scoped_claims(
    db: AsyncSession, cluster_ids: list, docs: set[uuid.UUID]
) -> dict[uuid.UUID, list[dict]]:
    """Each cluster's claims from in-scope volumes, with their volume -- the material a
    scoped view is rebuilt from. Clusters with no claim rows yet are absent."""
    if not cluster_ids:
        return {}
    rows = (
        await db.execute(
            text(
                "select cc.cluster_id, cc.text, cc.document_id, d.title from cluster_claim cc"
                " join document d on d.id = cc.document_id"
                " where cc.cluster_id = any(:ids) and cc.document_id = any(:docs)"
                " order by cc.cluster_id, d.title, cc.claim_index"
            ),
            {"ids": list(cluster_ids), "docs": list(docs)},
        )
    ).all()
    # A cluster with claim rows but none in scope maps to an empty list, not to absent.
    out: dict[uuid.UUID, list[dict]] = {
        cid: []
        for cid in (
            await db.execute(
                text("select distinct cluster_id from cluster_claim where cluster_id = any(:ids)"),
                {"ids": list(cluster_ids)},
            )
        ).scalars()
    }
    for cid, claim, did, title in rows:
        out.setdefault(cid, []).append({"claim": claim, "document_id": did, "source": title})
    return out


def scope_thread(t: dict, claims: list[dict] | None, docs: set[uuid.UUID]) -> dict | None:
    """A thread as seen from inside a scope, or None when it is not a thread there.

    Its whole-library summary speaks for volumes outside the scope, so a scoped thread is
    rebuilt from its in-scope claims: those claims, their volumes, and a disagreement only
    when both of its sides are in scope. Fewer than two in-scope volumes is not a
    connection across volumes, and is dropped."""
    if claims is None:
        # No claim rows yet: fall back to the claims the summary lists, by source title.
        titles = {
            title
            for title, did in zip(t.get("documents") or [], t.get("_doc_ids") or [], strict=False)
            if did in docs
        }
        claims = [
            {"claim": c, "source": src, "document_id": None}
            for c, src in zip(t.get("_listed") or [], t.get("_listed_sources") or [], strict=False)
            if src in titles
        ]
    in_docs = {c["source"] for c in claims}
    if len(in_docs) < 2:
        return None
    out = dict(t)
    out["documents"] = sorted(in_docs)
    out["scoped"] = {
        "documents": len(in_docs),
        "of_documents": t.get("document_count") or len(t.get("documents") or []),
    }
    # Spread across volumes: one claim from each in turn, so no single volume fills it.
    by_doc: dict[str, list[dict]] = {}
    for c in claims:
        by_doc.setdefault(c["source"], []).append(c)
    picked: list[dict] = []
    while len(picked) < SCOPED_CLAIMS and any(by_doc.values()):
        for src in list(by_doc):
            if by_doc[src] and len(picked) < SCOPED_CLAIMS:
                picked.append(by_doc[src].pop(0))
    out["claims"] = [{"source": c["source"], "claim": c["claim"]} for c in picked]
    cx = t.get("contradiction")
    if cx and not (cx["a"]["source"] in in_docs and cx["b"]["source"] in in_docs):
        out["contradiction"] = None
    return out


_CONTRA = """
select a.data->>'claim_a' as claim_a, a.data->>'source_a' as source_a,
       a.data->>'claim_b' as claim_b, a.data->>'source_b' as source_b, a.text as explanation
from artifact a
where a.kind = 'contradiction' and a.target_id = any(cast(:ids as uuid[]))
"""


async def relevant_threads(
    db: AsyncSession,
    brief: str,
    *,
    client=None,
    limit: int = 10,
    category_ids: list[uuid.UUID] | None = None,
    cartridge_ids: list[uuid.UUID] | None = None,
    levels: list[str] | None = None,
) -> list[dict]:
    """Cross-document threads nearest the brief, each with its documents and, where the
    library judged one, the disagreement at its heart."""
    if limit <= 0:
        return []
    vec = await embed_query(brief, client)
    docs = await scope_documents(db, category_ids, cartridge_ids, levels)
    if docs is not None and not docs:
        return []
    rows = (
        await db.execute(
            text(_SQL),
            {
                "qv": str(vec),
                "emodel": settings().embed_model,
                "docs": [str(x) for x in docs] if docs is not None else None,
                # A scope drops threads with one in-scope volume, so look further for them.
                "lim": limit * 3 if docs is not None else limit,
            },
        )
    ).all()
    threads = [
        {
            "id": r.id,
            "label": r.label,
            "summary": r.summary,
            "documents": list(r.documents or []),
            "document_count": r.document_count,
            "has_contradiction": r.has_contradiction,
            "contradiction": None,
            "_doc_ids": list(r.document_ids or []),
            "_listed": list(r.listed or []),
            "_listed_sources": list(r.listed_sources or []),
        }
        for r in rows
    ]
    disputed = [t["id"] for t in threads if t["has_contradiction"]]
    if disputed:
        cx = {
            r[0]: r
            for r in (
                await db.execute(
                    text(
                        "select a.target_id, a.data->>'claim_a' as claim_a, "
                        "a.data->>'source_a' as source_a, a.data->>'claim_b' as claim_b, "
                        "a.data->>'source_b' as source_b, a.text as explanation "
                        "from artifact a where a.kind = 'contradiction' "
                        "and a.target_id = any(cast(:ids as uuid[]))"
                    ),
                    {"ids": [str(x) for x in disputed]},
                )
            ).all()
        }
        for t in threads:
            r = cx.get(t["id"])
            if r and r.claim_a and r.claim_b:
                t["contradiction"] = {
                    "a": {"source": r.source_a or "", "claim": r.claim_a},
                    "b": {"source": r.source_b or "", "claim": r.claim_b},
                    "explanation": r.explanation or "",
                }
    if docs is not None:
        claims = await scoped_claims(db, [t["id"] for t in threads], docs)
        threads = [
            s
            for t in threads
            if (s := scope_thread(t, claims.get(t["id"]) if t["id"] in claims else None, docs))
        ][:limit]
    for t in threads:
        for k in ("_doc_ids", "_listed", "_listed_sources"):
            t.pop(k, None)
    return threads


def render_threads(threads: list[dict], *, max_chars: int = 6000) -> str:
    """The threads as a block for the planning prompt: what the library already connects,
    and where it found the works at odds."""
    if not threads:
        return ""
    lines: list[str] = []
    for t in threads:
        docs = ", ".join(t["documents"][:6])
        if t.get("scoped"):
            # Inside a scope the thread is its in-scope claims; the library-wide summary
            # would speak for volumes the document may not use.
            sc = t["scoped"]
            lines.append(
                f"- {t['label']} ({sc['documents']} of its {sc['of_documents']} volumes are "
                f"in scope; what they claim):"
            )
            lines += [f"    · {c['source']}: {c['claim']}" for c in t.get("claims", [])]
        else:
            head = f"- {t['label']}: {(t['summary'] or '').strip()}"
            if docs:
                head += f" (across: {docs})"
            lines.append(head)
        if c := t.get("contradiction"):
            lines.append(
                f"    ⚡ the library found these at odds — {c['a']['source']}: {c['a']['claim']} "
                f"vs {c['b']['source']}: {c['b']['claim']}"
            )
    out = "\n".join(lines)
    return out[:max_chars]
