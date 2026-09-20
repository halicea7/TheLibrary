"""Held passages lead the context; retrieval fills around them."""

from conftest import BODY_A, BODY_B, make_document
from sqlalchemy import select

from library_agent.chat.citations import build_sources
from library_agent.db.models import Chunk
from library_agent.retrieval.pipeline import hits_for_chunks


async def test_hits_for_chunks_keeps_order_and_outscores_retrieval(db):
    a = await make_document(db, title="Alpha", body=BODY_A)
    b = await make_document(db, title="Beta", body=BODY_B)
    ca = (await db.execute(select(Chunk).where(Chunk.document_id == a.id))).scalars().first()
    cb = (await db.execute(select(Chunk).where(Chunk.document_id == b.id))).scalars().first()
    hits = await hits_for_chunks(db, [cb.id, ca.id])
    assert [h.document_title for h in hits] == ["Beta", "Alpha"]
    assert all(h.score > 1 for h in hits)  # above any fused score
    srcs = build_sources(hits, {cb.id})
    assert [s.held for s in srcs] == [True, False]
    assert srcs[0].n == 1


async def test_unknown_chunk_ids_are_skipped(db):
    import uuid

    assert await hits_for_chunks(db, [uuid.uuid4()]) == []
