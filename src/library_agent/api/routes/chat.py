"""Chat endpoints. Answers stream over SSE."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from library_agent.chat.answer import STANCES
from library_agent.chat.session import get_or_create_conversation, run_turn
from library_agent.config import settings
from library_agent.db.models import Conversation, Message
from library_agent.db.session import SessionDep, session_scope
from library_agent.llm.ollama import Ollama

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    conversation_id: uuid.UUID | None = None
    conversational: bool = True
    model: str | None = None
    document_ids: list[uuid.UUID] | None = None
    category_ids: list[uuid.UUID] | None = None
    stance: str | None = None


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str | None
    conversational: bool
    model: str | None
    messages: int


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    sources: dict | None
    rewritten_query: str | None


@router.get("/chat/models")
async def chat_models() -> dict[str, object]:
    cfg = settings()
    async with Ollama() as c:
        thinking = {
            name: await c.supports_thinking(name) for name in cfg.chat_model_options.values()
        }
    return {
        "default": cfg.chat_model,
        "options": cfg.chat_model_options,
        # Which models can think. Thinking cannot be switched off cleanly on the ones that
        # can -- with think=false the model narrates its reasoning into the answer -- so the
        # client toggle is show/hide, never on/off.
        "thinking": thinking,
        "stances": [
            {"id": k, "label": v["label"], "counterpart": v.get("counterpart")}
            for k, v in STANCES.items()
        ],
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
        # Stored on the conversation so every turn is scoped the same way.
        conv.category_ids = [str(x) for x in req.category_ids] if req.category_ids else None
        conversation_id = conv.id

    async def events() -> AsyncIterator[dict]:
        yield {
            "event": "conversation",
            "data": json.dumps({"conversation_id": str(conversation_id)}),
        }
        async for ev in run_turn(
            conversation_id, req.message, document_ids=req.document_ids, stance=req.stance
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
        n = len(
            list(
                (
                    await db.execute(select(Message.id).where(Message.conversation_id == c.id))
                ).scalars()
            )
        )
        out.append(
            ConversationOut(
                id=c.id, title=c.title, conversational=c.conversational, model=c.model, messages=n
            )
        )
    return out


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
