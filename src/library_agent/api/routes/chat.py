"""Chat endpoints. Answers stream over SSE."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sse_starlette.sse import EventSourceResponse

from library_agent.chat.answer import STANCES
from library_agent.chat.session import get_or_create_conversation, run_turn
from library_agent.db.models import Conversation, Message
from library_agent.db.session import SessionDep, session_scope
from library_agent.library.shelving import expand_category_ids
from library_agent.llm import providers
from library_agent.llm.client import LLM

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    conversation_id: uuid.UUID | None = None
    conversational: bool = True
    model: str | None = None
    document_ids: list[uuid.UUID] | None = None
    category_ids: list[uuid.UUID] | None = None
    cartridge_ids: list[uuid.UUID] | None = None
    stance: str | None = None
    effort: str | None = None  # quick | normal | deep
    # Passages held from Find or Threads: they lead the context, retrieval fills around.
    pinned_chunk_ids: list[uuid.UUID] | None = None


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str | None
    conversational: bool
    model: str | None
    messages: int
    created_at: datetime | None = None
    last_at: datetime | None = None
    stance: str | None = None


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    sources: dict | None
    rewritten_query: str | None


@router.get("/chat/models")
async def chat_models() -> dict[str, object]:
    options = providers.chat_options()
    async with LLM() as c:
        thinking = {name: await c.supports_thinking(name) for name in options.values()}
    return {
        "default": options["general"],
        "options": options,
        # Which models can think. Thinking cannot be switched off cleanly on the ones that
        # can -- with think=false the model narrates its reasoning into the answer -- so the
        # client toggle is show/hide, never on/off.
        "thinking": thinking,
        "stances": [
            {"id": k, "label": v["label"], "counterpart": v.get("counterpart")}
            for k, v in STANCES.items()
        ],
        "efforts": ["quick", "normal", "deep"],
    }


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    if not req.message.strip():
        raise HTTPException(422, "message is empty")
    if req.stance and req.stance not in STANCES:
        raise HTTPException(422, f"unknown stance {req.stance!r}; one of {sorted(STANCES)}")

    async with session_scope() as db:
        conv = await get_or_create_conversation(
            db, req.conversation_id, conversational=req.conversational, model=req.model
        )
        if req.model and conv.model != req.model:
            conv.model = req.model
        if req.conversational != conv.conversational:
            conv.conversational = req.conversational
        if not conv.title:
            # The first question names the conversation; nothing cleverer is needed.
            conv.title = " ".join(req.message.split())[:90]
        # Stored on the conversation so every turn is scoped the same way.
        cat_ids = await expand_category_ids(db, req.category_ids or [])
        conv.category_ids = [str(x) for x in cat_ids] if cat_ids else None
        conv.cartridge_ids = [str(x) for x in req.cartridge_ids] if req.cartridge_ids else None
        conversation_id = conv.id

    async def events() -> AsyncIterator[dict]:
        yield {
            "event": "conversation",
            "data": json.dumps({"conversation_id": str(conversation_id)}),
        }
        async for ev in run_turn(
            conversation_id,
            req.message,
            document_ids=req.document_ids,
            stance=req.stance,
            effort=req.effort,
            pinned_chunk_ids=req.pinned_chunk_ids,
        ):
            payload = ev["data"]
            yield {
                "event": ev["event"],
                "data": payload if isinstance(payload, str) else json.dumps(payload),
            }

    return EventSourceResponse(events())


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(db: SessionDep, limit: int = 20) -> list[ConversationOut]:
    convs = list(
        (
            await db.execute(
                select(Conversation).order_by(Conversation.created_at.desc()).limit(limit)
            )
        ).scalars()
    )
    out = []
    for c in convs:
        n, last = (
            await db.execute(
                select(func.count(), func.max(Message.created_at)).where(
                    Message.conversation_id == c.id
                )
            )
        ).one()
        if not n:
            continue  # a conversation nobody spoke in is not worth listing
        last_stance = (
            await db.execute(
                select(Message.sources["stance"].astext)
                .where(Message.conversation_id == c.id, Message.role == "assistant")
                .order_by(Message.created_at.desc())
                .limit(1)
            )
        ).scalar()
        title = c.title
        if not title:
            # Conversations from before titles existed: name them after their first question.
            title = (
                await db.execute(
                    select(Message.content)
                    .where(Message.conversation_id == c.id, Message.role == "user")
                    .order_by(Message.created_at)
                    .limit(1)
                )
            ).scalar()
            title = " ".join((title or "").split())[:90] or None
        out.append(
            ConversationOut(
                id=c.id,
                title=title,
                conversational=c.conversational,
                model=c.model,
                messages=n,
                created_at=c.created_at,
                last_at=last,
                stance=last_stance,
            )
        )
    out.sort(key=lambda x: x.last_at or x.created_at, reverse=True)
    return out


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: uuid.UUID, db: SessionDep) -> dict[str, bool]:
    conv = await db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(404, "no such conversation")
    await db.delete(conv)  # messages cascade
    await db.commit()
    return {"deleted": True}


@router.get("/conversations/{conversation_id}", response_model=list[MessageOut])
async def get_conversation(conversation_id: uuid.UUID, db: SessionDep) -> list[MessageOut]:
    rows = list(
        (
            await db.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            )
        ).scalars()
    )
    return [
        MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            sources=m.sources,
            rewritten_query=m.rewritten_query,
        )
        for m in rows
    ]
