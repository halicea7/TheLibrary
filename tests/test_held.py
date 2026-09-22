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


class TestConversationOrder:
    """Both messages of a turn are written in one transaction and share a timestamp;
    the question must still come back before the answer."""

    async def test_a_turn_replays_question_first(self, db):
        from sqlalchemy import select

        from library_agent.db.models import Conversation, Message

        conv = Conversation(title="T")
        db.add(conv)
        await db.flush()
        # written answer-first, as the row order that caused the bug had them
        db.add(Message(conversation_id=conv.id, role="assistant", content="A", sources={"n": 1}))
        db.add(Message(conversation_id=conv.id, role="user", content="Q"))
        await db.flush()
        rows = list(
            (
                await db.execute(
                    select(Message)
                    .where(Message.conversation_id == conv.id)
                    .order_by(*Message.in_order())
                )
            ).scalars()
        )
        assert [m.role for m in rows] == ["user", "assistant"]
        assert [m.content for m in rows] == ["Q", "A"]
