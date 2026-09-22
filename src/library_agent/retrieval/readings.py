"""The library's own reading, as context for an answer.

Retrieval finds passages: a page's worth of the text at a time. For a question about what
a work says across its chapters -- "how would the book apply to X" -- the right material
is the library's summary of each section, which Tier 1 wrote and embedded and Ask never
used. These are searched the same way, in the same scope, and handed to the model as
numbered items beside the passages, cited like them, pointing at the section's first
passage so the reader opens where the reading begins.

And a question that names a volume is about that volume: `named_documents` finds it, so
retrieval can lift the per-volume cap for it rather than cutting the subject at three."""

from __future__ import annotations

import re
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.llm.embed import embed_query
from library_agent.retrieval.hybrid import SearchHit

# A title shorter than this is too likely to occur in a question by accident ("Notes",
# "Kernel"); one that is mostly a filename is not how anyone names a book.
MIN_TITLE = 10

_SQL = """
select s.document_id, d.title, coalesce(s.path, s.title, '') as path, s.page_start, a.text,
       (select c.id from chunk c where c.section_id = s.id
        order by c.order_index limit 1) as chunk_id,
       1 - (e.vec <=> cast(:qv as halfvec)) as score
from embedding e
join artifact a on a.id = e.owner_id and a.kind = 'section_summary'
join section s on s.id = a.target_id
join document d on d.id = s.document_id
where e.owner_kind = 'artifact' and e.model = :emodel
  and (cast(:docs as uuid[]) is null or s.document_id = any(cast(:docs as uuid[])))
  and (cast(:cats as uuid[]) is null or exists (
      select 1 from document_category dc
      where dc.document_id = s.document_id and dc.category_id = any(cast(:cats as uuid[]))
  ) or exists (
      select 1 from chunk c join chunk_category cc on cc.chunk_id = c.id
      where c.section_id = s.id and cc.category_id = any(cast(:cats as uuid[]))
  ))
  and (cast(:carts as uuid[]) is null or exists (
      select 1 from cartridge_document cd
      where cd.document_id = s.document_id and cd.cartridge_id = any(cast(:carts as uuid[]))
  ))
order by e.vec <=> cast(:qv as halfvec)
limit :lim
"""


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return " " + re.sub(r"\s+", " ", s).strip() + " "


async def named_documents(db: AsyncSession, question: str) -> list[uuid.UUID]:
    """Volumes the question names by title, a leading "The" optional."""
    q = _norm(question)
    rows = (
        await db.execute(
            text("select id, title from document where length(title) >= :n"), {"n": MIN_TITLE}
        )
    ).all()
    out = []
    for did, title in rows:
        t = _norm(title)
        bare = re.sub(r"^ (the|a|an) ", " ", t)
        if len(bare.strip()) >= MIN_TITLE and (t in q or bare in q):
            out.append(did)
    return out


async def retrieve_readings(
    db: AsyncSession,
    query: str,
    *,
    client=None,
    limit: int,
    category_ids: list[uuid.UUID] | None = None,
    cartridge_ids: list[uuid.UUID] | None = None,
    document_ids: list[uuid.UUID] | None = None,
    favour: list[uuid.UUID] | None = None,
) -> list[SearchHit]:
    """Section readings nearest the query. With `favour` (named volumes), most of the
    budget goes to their sections and the rest to the scope at large."""
    if limit <= 0:
        return []
    vec = await embed_query(query, client)
    base = {
        "qv": str(vec),
        "emodel": settings().embed_model,
        "cats": [str(x) for x in category_ids] if category_ids else None,
        "carts": [str(x) for x in cartridge_ids] if cartridge_ids else None,
    }

    async def run(docs, lim):
        rows = (
            await db.execute(
                text(_SQL),
                {**base, "docs": [str(d) for d in docs] if docs else None, "lim": lim},
            )
        ).all()
        return [
            SearchHit(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_title=r.title,
                section_path=r.path,
                page=r.page_start,
                text=r.text,
                score=float(r.score),
                dense_rank=None,
                lexical_rank=None,
                kind="reading",
            )
            for r in rows
            if r.chunk_id is not None
        ]

    named = [d for d in (favour or []) if not document_ids or d in set(document_ids)]
    if not named:
        return await run(document_ids, limit)
    own = await run(named, max(1, limit - limit // 4))
    rest = await run(document_ids, limit)
    seen = {(h.document_id, h.section_path) for h in own}
    return (own + [h for h in rest if (h.document_id, h.section_path) not in seen])[:limit]
