"""Duplicate detection.

Exact content hashing catches re-uploads of the same bytes. It does not catch the case
that actually happens in a paper library: arXiv v1 and v2, or the same PDF obtained from
two sources, which differ by a timestamp but are the same document."""

from __future__ import annotations

import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import Document, OwnerKind

# Cosine similarity above which two documents are treated as the same work.
NEAR_DUP_THRESHOLD = 0.93

FINGERPRINT_CHARS = 4000


def fingerprint_text(title: str, body: str) -> str:
    """Title plus opening text. The opening of a paper (title/abstract/intro) identifies
    it far more reliably than a sample from the middle."""
    return f"{title}\n\n{body[:FINGERPRINT_CHARS]}".strip()


async def find_exact(db: AsyncSession, content_hash: str) -> Document | None:
    return (
        await db.execute(select(Document).where(Document.content_hash == content_hash))
    ).scalar_one_or_none()


async def find_near_duplicate(
    db: AsyncSession, vec: list[float], exclude_id: uuid.UUID | None = None
) -> tuple[uuid.UUID, float] | None:
    """Nearest existing document fingerprint, if it is close enough to be the same work."""
    sql = """
        select e.owner_id, 1 - (e.vec <=> (:v)::halfvec) as sim
        from embedding e
        where e.owner_kind = :k and e.model = :m
          and (cast(:excl as uuid) is null or e.owner_id <> cast(:excl as uuid))
        order by e.vec <=> (:v)::halfvec
        limit 1
    """
    row = (
        await db.execute(
            text(sql),
            {
                "v": str(vec),
                "k": OwnerKind.DOCUMENT.value,
                "m": settings().embed_model,
                "excl": str(exclude_id) if exclude_id else None,
            },
        )
    ).first()
    if row and row.sim >= NEAR_DUP_THRESHOLD:
        return row.owner_id, float(row.sim)
    return None
