"""Document-level routing.

At 500 heterogeneous documents, the top chunks for any query come from a dozen unrelated
documents that share vocabulary, and the right passage sits below the fold. Scoring the
query against document-level vectors first (a few hundred rows, sub-millisecond) narrows
the chunk search to plausible documents and keeps retrieval cost flat as the corpus grows.

A router miss would otherwise be unrecoverable, so part of the candidate budget always
searches the whole corpus unrouted."""

from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import ArtifactKind, OwnerKind

# Prefer a real document summary (Tier 1) but fall back to the Tier 0 fingerprint, so
# routing works from the moment a document is ingested.
_SQL = """
with doc_vectors as (
    select d.id as document_id,
           coalesce(summary_emb.vec, fp.vec) as vec
    from document d
    left join embedding fp
        on fp.owner_kind = :dockind and fp.owner_id = d.id and fp.model = :emodel
    left join lateral (
        select e.vec
        from artifact a
        join embedding e on e.owner_kind = :artkind and e.owner_id = a.id and e.model = :emodel
        where a.target_id = d.id and a.kind = :summary_kind
        limit 1
    ) summary_emb on true
    where coalesce(summary_emb.vec, fp.vec) is not null
)
select document_id, 1 - (vec <=> cast(:qv as halfvec)) as sim
from doc_vectors
order by vec <=> cast(:qv as halfvec)
limit :k
"""


async def route(
    db: AsyncSession, query_vector: list[float], *, top_k: int | None = None
) -> list[tuple[uuid.UUID, float]]:
    cfg = settings()
    rows = (
        await db.execute(
            text(_SQL),
            {
                "qv": str(query_vector),
                "emodel": cfg.embed_model,
                "dockind": OwnerKind.DOCUMENT.value,
                "artkind": OwnerKind.ARTIFACT.value,
                "summary_kind": ArtifactKind.DOCUMENT_SUMMARY.value,
                "k": top_k or cfg.router_top_documents,
            },
        )
    ).all()
    return [(r.document_id, float(r.sim)) for r in rows]
