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
       (select array_agg(distinct d.title)
        from cluster_member m join document d on d.id = m.document_id
        where m.cluster_id = cl.id) as documents
from embedding e
join artifact a on a.id = e.owner_id and a.kind = 'cluster_summary'
join cluster cl on cl.id = a.target_id
where e.owner_kind = 'artifact' and e.model = :emodel
  and cl.document_count > 1
  and (cast(:cats as uuid[]) is null or exists (
      select 1 from cluster_member m join document_category dc on dc.document_id = m.document_id
      where m.cluster_id = cl.id and dc.category_id = any(cast(:cats as uuid[]))
  ))
  and (cast(:carts as uuid[]) is null or exists (
      select 1 from cluster_member m join cartridge_document cd on cd.document_id = m.document_id
      where m.cluster_id = cl.id and cd.cartridge_id = any(cast(:carts as uuid[]))
  ))
order by e.vec <=> cast(:qv as halfvec)
limit :lim
"""

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
) -> list[dict]:
    """Cross-document threads nearest the brief, each with its documents and, where the
    library judged one, the disagreement at its heart."""
    if limit <= 0:
        return []
    vec = await embed_query(brief, client)
    rows = (
        await db.execute(
            text(_SQL),
            {
                "qv": str(vec),
                "emodel": settings().embed_model,
                "cats": [str(x) for x in category_ids] if category_ids else None,
                "carts": [str(x) for x in cartridge_ids] if cartridge_ids else None,
                "lim": limit,
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
    return threads


def render_threads(threads: list[dict], *, max_chars: int = 6000) -> str:
    """The threads as a block for the planning prompt: what the library already connects,
    and where it found the works at odds."""
    if not threads:
        return ""
    lines: list[str] = []
    for t in threads:
        docs = ", ".join(t["documents"][:6])
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
