"""The library for other programs.

Plain JSON, no streaming, stable shapes. `ask` collects a whole chat turn -- the same
retrieval, generation and citation verification the UI gets -- and returns the answer
with its citations resolved. Rooms and subjects are addressed by *name*, because a tool
should be able to say "ask our docs" without first learning our ids."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from library_agent.api.routes.documents import read_document as _read_document
from library_agent.chat.answer import STANCES
from library_agent.chat.session import get_or_create_conversation, run_turn
from library_agent.config import settings
from library_agent.db.models import Cartridge, CartridgeDocument, Category, Document
from library_agent.db.session import SessionDep, session_scope
from library_agent.library.shelving import expand_category_ids
from library_agent.retrieval.pipeline import RetrievalConfig, retrieve

router = APIRouter(prefix="/api/v1", tags=["v1"])


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    room: str | None = Field(default=None, description="a cartridge, by name or id")
    subjects: list[str] = Field(default_factory=list, description="shelf names to stay within")
    model: str | None = Field(default=None, description="'general' or 'technical', or a model name")
    stance: str | None = None
    conversation_id: uuid.UUID | None = None
    remember: bool = Field(default=True, description="keep the thread for follow-ups")


class Citation(BaseModel):
    n: int
    title: str
    page: int | None
    section: str | None
    document_id: str
    cartridge: dict | None = None
    readings_only: bool = False


class AskOut(BaseModel):
    answer: str
    citations: list[Citation]
    conversation_id: str
    model: str
    verified: dict
    room: str | None = None


async def _room_id(db, room: str | None) -> uuid.UUID | None:
    if not room:
        return None
    try:
        rid = uuid.UUID(room)
        if await db.get(Cartridge, rid):
            return rid
    except ValueError:
        pass
    row = (
        (await db.execute(select(Cartridge).where(func.lower(Cartridge.name) == room.lower())))
        .scalars()
        .first()
    )
    if not row:
        raise HTTPException(404, f"no cartridge named {room!r}")
    return row.id


async def _subject_ids(db, names: list[str]) -> list[uuid.UUID]:
    if not names:
        return []
    rows = list(
        (
            await db.execute(
                select(Category).where(func.lower(Category.name).in_([n.lower() for n in names]))
            )
        ).scalars()
    )
    missing = {n.lower() for n in names} - {r.name.lower() for r in rows}
    if missing:
        raise HTTPException(404, f"no shelf named {sorted(missing)}")
    return await expand_category_ids(db, [r.id for r in rows])


def _model(name: str | None) -> str | None:
    cfg = settings()
    if not name:
        return None
    return cfg.chat_model_options.get(name, name)


@router.post("/ask", response_model=AskOut)
async def ask(req: AskIn) -> AskOut:
    """One whole turn, as JSON. Every `[n]` in the answer is a citation below; anything
    the model cited that could not be verified against the shelf has been stripped."""
    if req.stance and req.stance not in STANCES:
        raise HTTPException(422, f"unknown stance {req.stance!r}; one of {sorted(STANCES)}")
    async with session_scope() as db:
        room = await _room_id(db, req.room)
        subjects = await _subject_ids(db, req.subjects)
        conv = await get_or_create_conversation(
            db, req.conversation_id, conversational=req.remember, model=_model(req.model)
        )
        if req.model:
            conv.model = _model(req.model)
        if not conv.title:
            conv.title = " ".join(req.question.split())[:90]
        conv.category_ids = [str(x) for x in subjects] or None
        conv.cartridge_ids = [str(room)] if room else None
        conversation_id = conv.id

    answer, sources, model, done, error = "", [], "", {}, None
    async for ev in run_turn(conversation_id, req.question, stance=req.stance):
        kind, data = ev["event"], ev["data"]
        if kind == "meta":
            model = data.get("model", "")
        elif kind == "sources":
            sources = data
        elif kind == "done":
            answer, done = data["answer"], data
        elif kind == "error":
            error = data
    if error:
        raise HTTPException(502, error)
    cited = set(done.get("cited") or [])
    return AskOut(
        answer=answer,
        citations=[
            Citation(
                n=s["n"],
                title=s["title"],
                page=s.get("page"),
                section=s.get("section"),
                document_id=s["document_id"],
                cartridge=s.get("cartridge"),
                readings_only=bool(s.get("readings_only")),
            )
            for s in sources
            if s["n"] in cited
        ],
        conversation_id=str(conversation_id),
        model=model,
        verified={
            "emitted": done.get("markers_emitted", 0),
            "resolved": done.get("markers_resolved", 0),
        },
        room=req.room,
    )


@router.get("/search")
async def search(
    q: str, db: SessionDep, room: str | None = None, subjects: str | None = None, limit: int = 8
) -> dict:
    """Passages, ranked. `subjects` is comma-separated shelf names."""
    if len(q.strip()) < 2:
        raise HTTPException(422, "q is too short")
    rid = await _room_id(db, room)
    sids = await _subject_ids(db, [s for s in (subjects or "").split(",") if s.strip()])
    hits = await retrieve(
        db,
        q,
        limit=max(1, min(50, limit)),
        category_ids=sids or None,
        cartridge_ids=[rid] if rid else None,
        config=RetrievalConfig(name="v1", use_reranker=True, use_router=False),
    )
    return {
        "query": q,
        "room": room,
        "hits": [
            {
                "document_id": str(h.document_id),
                "title": h.document_title,
                "page": h.page,
                "section": h.section_path,
                "text": h.text,
                "score": h.score,
            }
            for h in hits
        ],
    }


@router.get("/shelves")
async def shelves(db: SessionDep) -> dict:
    """What there is to ask about: top shelves with their sub-shelves and counts, and the
    cartridges on the rack (each a room you can ask by name)."""
    cats = list((await db.execute(select(Category))).scalars())
    by_id = {c.id: c for c in cats}
    shelved = dict(
        (
            await db.execute(
                select(Document.shelf_id, func.count())
                .where(Document.shelf_id.is_not(None))
                .group_by(Document.shelf_id)
            )
        ).all()
    )
    tops: dict[str, list] = {}
    for c in cats:
        if c.parent_id and c.parent_id in by_id:
            tops.setdefault(by_id[c.parent_id].name, []).append(
                {"name": c.name, "volumes": shelved.get(c.id, 0)}
            )
    rooms = [
        {
            "name": c.name,
            "id": str(c.id),
            "level": c.level,
            "clearance": (c.design or {}).get("clearance", "open"),
            "volumes": n,
        }
        for c, n in (
            await db.execute(
                select(Cartridge, func.count(CartridgeDocument.document_id))
                .outerjoin(CartridgeDocument, CartridgeDocument.cartridge_id == Cartridge.id)
                .group_by(Cartridge.id)
            )
        ).all()
    ]
    total = (await db.execute(select(func.count()).select_from(Document))).scalar_one()
    return {
        "volumes": total,
        "shelves": [
            {
                "name": t,
                "volumes": sum(s["volumes"] for s in subs),
                "sub_shelves": sorted(subs, key=lambda s: -s["volumes"]),
            }
            for t, subs in sorted(tops.items(), key=lambda kv: -sum(s["volumes"] for s in kv[1]))
        ],
        "rooms": rooms,
    }


@router.get("/volumes/{document_id}")
async def volume(document_id: uuid.UUID, db: SessionDep) -> dict:
    """A volume as the reader sees it: sections, passages, and what the library wrote."""
    return await _read_document(document_id, db)
