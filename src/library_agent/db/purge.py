"""Deletion for polymorphic rows.

`embedding.owner_id` and `artifact.target_id` intentionally carry no foreign key -- they
point at chunks, sections, documents or clusters depending on their kind. The cost of
that flexibility is that Postgres cannot cascade them, so deleting a document would
silently orphan every vector and artifact it owned. This module is the only correct way
to remove a document."""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_PURGE_VECTORS_AND_ARTIFACTS = """
with ids as (
    select cast(:doc as uuid) as document_id
),
chunk_ids as (select id from chunk where document_id = cast(:doc as uuid)),
section_ids as (select id from section where document_id = cast(:doc as uuid)),
artifact_ids as (
    select a.id from artifact a
    where a.target_id = cast(:doc as uuid)
       or a.target_id in (select id from chunk_ids)
       or a.target_id in (select id from section_ids)
),
del_emb as (
    delete from embedding e
    where (e.owner_kind = 'document' and e.owner_id = cast(:doc as uuid))
       or (e.owner_kind = 'chunk'    and e.owner_id in (select id from chunk_ids))
       or (e.owner_kind = 'artifact' and e.owner_id in (select id from artifact_ids))
    returning 1
),
del_art as (
    delete from artifact where id in (select id from artifact_ids) returning 1
)
select (select count(*) from del_emb) as embeddings,
       (select count(*) from del_art) as artifacts
"""


async def purge_document_dependents(db: AsyncSession, document_id: uuid.UUID) -> tuple[int, int]:
    """Remove vectors and artifacts belonging to a document. Call BEFORE deleting the
    document row, while its chunks and sections are still resolvable."""
    row = (await db.execute(text(_PURGE_VECTORS_AND_ARTIFACTS), {"doc": str(document_id)})).first()
    return (row.embeddings, row.artifacts) if row else (0, 0)


async def delete_document(db: AsyncSession, document_id: uuid.UUID) -> tuple[int, int]:
    """Full removal: dependents first, then the row (chunks/sections cascade normally),
    then the stored file.

    The store is content-addressed -- one file per content hash, one document row per
    hash -- so deleting the row means nothing references the file. Leaving it behind is
    how a quarantined test string (EICAR, from a payload collection) stayed on disk after
    its document was gone and tripped the endpoint agent."""
    counts = await purge_document_dependents(db, document_id)
    path = (
        await db.execute(
            text("select source_path from document where id = :d"), {"d": str(document_id)}
        )
    ).scalar()
    await db.execute(text("delete from document where id = :d"), {"d": str(document_id)})
    if path:
        from library_agent.config import settings

        target = Path(path)
        # Only ever remove something inside the store; a row pointing elsewhere is a bug,
        # not a licence.
        if target.resolve().is_relative_to(settings().storage_dir.resolve()):
            target.unlink(missing_ok=True)
    return counts


def gc_store(referenced_hashes: set[str]) -> int:
    """Remove stored files no document references. Safe to run any time."""
    from library_agent.config import settings

    removed = 0
    store = settings().storage_dir
    if not store.exists():
        return 0
    for f in store.rglob("*"):
        if f.is_file() and f.stem not in referenced_hashes:
            f.unlink(missing_ok=True)
            removed += 1
    return removed


async def count_orphans(db: AsyncSession) -> dict[str, int]:
    """Health check for the invariant this module exists to maintain."""
    sql = """
    select
      (select count(*) from embedding e where e.owner_kind='chunk'
         and not exists (select 1 from chunk c where c.id = e.owner_id)) as orphan_chunk_vectors,
      (select count(*) from embedding e where e.owner_kind='document'
         and not exists (select 1 from document d where d.id = e.owner_id)) as orphan_doc_vectors,
      (select count(*) from embedding e where e.owner_kind='artifact'
         and not exists (select 1 from artifact a where a.id = e.owner_id)) as orphan_artifact_vectors
    """
    r = (await db.execute(text(sql))).first()
    return dict(r._mapping) if r else {}


_GC = """
with dead as (
    delete from embedding e
    where (e.owner_kind = 'chunk'
             and not exists (select 1 from chunk c where c.id = e.owner_id))
       or (e.owner_kind = 'document'
             and not exists (select 1 from document d where d.id = e.owner_id))
       or (e.owner_kind = 'artifact'
             and not exists (select 1 from artifact a where a.id = e.owner_id))
    returning 1
)
select count(*) as removed from dead
"""


async def gc_orphan_vectors(db: AsyncSession) -> int:
    """Sweep vectors whose owner no longer exists.

    delete_document() keeps this at zero, but a crash between statements -- or rows
    written before that path existed -- can still leave debris. Safe to run any time."""
    row = (await db.execute(text(_GC))).first()
    return int(row.removed) if row else 0
