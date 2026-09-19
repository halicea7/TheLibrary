"""Chat orchestration: rewrite → retrieve → generate → validate → persist."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.chat import answer as answer_mod
from library_agent.chat.citations import Source, build_sources, citation_validity, validate
from library_agent.chat.rewrite import rewrite_query
from library_agent.config import settings
from library_agent.db.models import Conversation, Document, Message
from library_agent.db.session import session_scope
from library_agent.llm.lease import mark_chat_active, mark_chat_done, redis_client
from library_agent.llm.ollama import Ollama
from library_agent.retrieval.hybrid import SearchHit
from library_agent.retrieval.pipeline import CHAT_RETRIEVAL, retrieve

log = logging.getLogger(__name__)


@dataclass
class TurnState:
    question: str
    rewritten: str | None = None
    hits: list[SearchHit] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    answer: str = ""


async def _history(db: AsyncSession, conversation_id: uuid.UUID) -> list[tuple[str, str]]:
    rows = (
        await db.execute(
            select(Message.role, Message.content)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
    ).all()
    return [(r.role, r.content) for r in rows]


async def get_or_create_conversation(
    db: AsyncSession,
    conversation_id: uuid.UUID | None,
    *,
    conversational: bool = True,
    model: str | None = None,
) -> Conversation:
    if conversation_id:
        conv = (
            await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        ).scalar_one_or_none()
        if conv:
            return conv
    conv = Conversation(conversational=conversational, model=model)
    db.add(conv)
    await db.flush()
    return conv


async def run_turn(
    conversation_id: uuid.UUID,
    question: str,
    *,
    document_ids: list[uuid.UUID] | None = None,
    stance: str | None = None,
) -> AsyncIterator[dict]:
    """Yields SSE-shaped events: meta → sources → thinking* → token* → done."""
    cfg = settings()
    redis = redis_client()
    client = Ollama()
    state = TurnState(question=question)

    try:
        # The lease is taken for the whole turn -- retrieval and generation both compete
        # with background reading for the same models.
        await mark_chat_active(redis)

        async with session_scope() as db:
            conv = (
                await db.execute(select(Conversation).where(Conversation.id == conversation_id))
            ).scalar_one()
            model = conv.model or cfg.chat_model
            conversational = conv.conversational
            category_ids = [uuid.UUID(x) for x in conv.category_ids] if conv.category_ids else None
            history = await _history(db, conversation_id) if conversational else []

        query = question
        needs_retrieval = True
        if conversational and history:
            query, needs_retrieval = await rewrite_query(client, model, question, history)
            state.rewritten = query if query != question else None

        yield {
            "event": "meta",
            "data": {
                "model": model,
                "rewritten": state.rewritten,
                "retrieving": needs_retrieval,
                "categories": len(category_ids) if category_ids else 0,
                "stance": stance,
            },
        }

        if needs_retrieval:
            async with session_scope() as db:
                state.hits = await retrieve(
                    db,
                    query,
                    config=CHAT_RETRIEVAL,
                    client=client,
                    limit=cfg.chat_passages,
                    category_ids=category_ids,
                )
                if document_ids:
                    state.hits = [h for h in state.hits if h.document_id in set(document_ids)]
                state.sources = build_sources(state.hits)
                titles = {
                    d.id: d.title
                    for d in (
                        await db.execute(
                            select(Document).where(
                                Document.id.in_(
                                    [h.document_id for h in state.hits] or [uuid.uuid4()]
                                )
                            )
                        )
                    ).scalars()
                }
            for s in state.sources:
                s.document_title = titles.get(uuid.UUID(s.document_id), s.document_title)

        yield {
            "event": "sources",
            "data": [
                {
                    "n": s.n,
                    "title": s.document_title,
                    "page": s.page,
                    "section": s.section_path,
                    "document_id": s.document_id,
                }
                for s in state.sources
            ],
        }

        messages = answer_mod.build_messages(
            question, state.hits, state.sources, history, stance=stance
        )
        # A stance is an invitation to interpret; give the sampler room to take it.
        temperature = 0.85 if stance else 0.6
        buffer: list[str] = []

        # Prefill on a local model can exceed the lease TTL before the first token
        # arrives, and a per-token refresh does nothing during that window -- a reading
        # job could resume mid-generation. Refresh on a timer instead.
        async def keepalive() -> None:
            while True:
                await asyncio.sleep(cfg.chat_lease_ttl_seconds / 3)
                await mark_chat_active(redis)

        heartbeat = asyncio.create_task(keepalive())
        try:
            async for kind, piece in answer_mod.stream_answer(
                client, model, messages, temperature=temperature
            ):
                if kind == "thinking":
                    yield {"event": "thinking", "data": piece}
                    continue
                buffer.append(piece)
                yield {"event": "token", "data": piece}
        finally:
            heartbeat.cancel()

        raw = "".join(buffer)
        cleaned, used = validate(raw, state.sources)
        metrics = citation_validity(raw, state.sources)
        state.answer = cleaned

        async with session_scope() as db:
            db.add(
                Message(
                    conversation_id=conversation_id,
                    role="user",
                    content=question,
                    rewritten_query=state.rewritten,
                )
            )
            db.add(
                Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=cleaned,
                    sources={
                        "cited": [
                            {
                                "n": s.n,
                                "title": s.document_title,
                                "page": s.page,
                                "chunk_id": s.chunk_id,
                                "document_id": s.document_id,
                            }
                            for s in used
                        ],
                        "retrieved": len(state.sources),
                        "stance": stance,
                        **metrics,
                    },
                )
            )

        yield {
            "event": "done",
            "data": {
                "answer": cleaned,
                "cited": [s.n for s in used],
                **metrics,
            },
        }
    except Exception as exc:
        log.exception("chat turn failed")
        yield {"event": "error", "data": str(exc)[:500]}
    finally:
        await client.aclose()
        await mark_chat_done(redis)
        await redis.aclose()
