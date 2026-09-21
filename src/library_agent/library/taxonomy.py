"""Category assignment.

The spec proposed categories per document and merged duplicates later with a scheduled
job. Steering at creation time is far cheaper: the canonical list goes into the tagging
prompt, so the model reuses "Information Retrieval" instead of inventing "IR", "Retrieval"
and "Document Search". The merge job stays as a backstop, not the primary mechanism."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.db.models import Category, ChunkCategory, DocumentCategory


def normalize_name(name: str) -> str:
    words = " ".join(name.strip().split()).strip(" .,:;").split(" ")
    # Title-case, but leave short all-caps tokens alone: NLP, IR, LLM, API, OSCP.
    return " ".join(w if (w.isupper() and 2 <= len(w) <= 5) else w.title() for w in words)


async def designate(db: AsyncSession, name: str) -> Category | None:
    """A name from a shelf design, taken literally. `get_or_create` follows a folded
    name to its survivor -- right for a tag, wrong for a design: "Security" once
    folded into "Network Services" came back as a top shelf called Network Services,
    beside a sub-shelf of the same name. Here the name is revived if it was folded,
    made if it is new, and never redirected."""
    clean = normalize_name(name)
    if len(clean) < 3 or len(clean) > 60:
        return None
    existing = (
        await db.execute(select(Category).where(Category.name == clean))
    ).scalar_one_or_none()
    if existing:
        existing.canonical = True
        return existing
    row = Category(name=clean, canonical=True)
    db.add(row)
    await db.flush()
    return row


async def canonical_names(db: AsyncSession) -> list[str]:
    return list(
        (await db.execute(select(Category.name).where(Category.canonical.is_(True)))).scalars()
    )


async def get_or_create(db: AsyncSession, name: str) -> Category | None:
    clean = normalize_name(name)
    if len(clean) < 3 or len(clean) > 60:
        return None
    existing = (
        await db.execute(select(Category).where(Category.name == clean))
    ).scalar_one_or_none()
    if existing and existing.canonical:
        return existing
    if existing:
        # Folded into another category: hand back the survivor, or revive this one if
        # the survivor is gone.
        survivor = (
            (
                await db.execute(
                    select(Category).where(
                        Category.canonical.is_(True),
                        Category.merged_from.contains([clean]),  # type: ignore[arg-type]
                    )
                )
            )
            .scalars()
            .first()
        )
        if survivor:
            return survivor
        existing.canonical = True
        return existing
    # A name previously folded into another category resolves to its survivor.
    merged = (
        (
            await db.execute(
                select(Category).where(Category.merged_from.contains([clean]))  # type: ignore[arg-type]
            )
        )
        .scalars()
        .first()
    )
    if merged:
        return merged
    row = Category(name=clean)
    db.add(row)
    await db.flush()
    return row


async def assign_document(
    db: AsyncSession, document_id: uuid.UUID, names: list[str], *, replace: bool = True
) -> list[Category]:
    if replace:
        await db.execute(
            delete(DocumentCategory).where(DocumentCategory.document_id == document_id)
        )
    out: list[Category] = []
    seen: set[uuid.UUID] = set()
    for n in names:
        cat = await get_or_create(db, n)
        if cat and cat.id not in seen:
            seen.add(cat.id)
            db.add(DocumentCategory(document_id=document_id, category_id=cat.id))
            out.append(cat)
    return out


async def assign_chunks(
    db: AsyncSession, chunk_ids: list[uuid.UUID], names: list[str]
) -> list[Category]:
    """Section-level labels applied to that section's chunks, so a chapter whose topic
    diverges from the document as a whole is still findable on its own terms."""
    if not chunk_ids:
        return []
    await db.execute(delete(ChunkCategory).where(ChunkCategory.chunk_id.in_(chunk_ids)))
    cats: list[Category] = []
    for n in names:
        cat = await get_or_create(db, n)
        if cat and cat not in cats:
            cats.append(cat)
    for cid in chunk_ids:
        for cat in cats:
            db.add(ChunkCategory(chunk_id=cid, category_id=cat.id))
    return cats
