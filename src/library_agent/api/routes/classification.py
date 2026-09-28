"""Classification: the scale, and a volume's level set by hand."""

from __future__ import annotations

import uuid
from dataclasses import replace

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from library_agent import classification as cls
from library_agent.db.models import Document
from library_agent.db.session import SessionDep, session_scope

router = APIRouter(prefix="/api", tags=["classification"])


class LevelIn(BaseModel):
    id: str
    label: str
    short: str
    colour: str = "#777777"
    banners: list[str] = []
    portions: list[str] = []


class ScaleIn(BaseModel):
    scheme: str | None = None  # "us" or "company" starts from that preset
    levels: list[LevelIn] | None = None
    default: str | None = None
    remote_ceiling: str | None = None
    export_ceiling: str | None = None
    mode: str | None = None  # strict | cited
    marking: bool | None = None


def _out(s: cls.Scale) -> dict:
    return {**s.public(), "presets": sorted(cls.PRESETS)}


@router.get("/settings/classification")
async def get_scale() -> dict:
    return _out(cls.load())


@router.put("/settings/classification")
async def put_scale(req: ScaleIn) -> dict:
    """Change the scale. Choosing a preset replaces the levels and ceilings; the library is
    then read again for markings, since a level id of the old scale means nothing in the new."""
    old = cls.load()
    s = cls.preset(req.scheme) if req.scheme and req.scheme != old.scheme else old
    if req.levels is not None:
        s = replace(s, scheme="custom", levels=[cls.Level(**lv.model_dump()) for lv in req.levels])
    for k in ("default", "remote_ceiling", "export_ceiling", "mode", "marking"):
        if (v := getattr(req, k)) is not None:
            s = replace(s, **{k: v})
    if s.mode not in ("strict", "cited"):
        raise HTTPException(422, "mode is strict or cited")
    try:
        cls.save(s)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    rescan = [lv.id for lv in s.levels] != [lv.id for lv in old.levels] or any(
        (a.banners, a.portions) != (b.banners, b.portions)
        for a, b in zip(s.levels, old.levels, strict=False)
    )
    out = _out(s)
    if rescan:
        out["rescanned"] = await cls.classify_all(session_scope)
    return out


@router.post("/settings/classification/rescan")
async def rescan() -> dict:
    """Read every volume again for banner lines and portion marks."""
    return await cls.classify_all(session_scope)


class DocLevelIn(BaseModel):
    level: str | None  # null: forget the hand-set level and read the markings again


@router.put("/documents/{document_id}/classification")
async def set_level(document_id: uuid.UUID, req: DocLevelIn, db: SessionDep) -> dict:
    s = cls.load()
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "no such volume")
    if req.level is None:
        doc.classification_source = None
        await cls.classify_document(db, doc, s)
    else:
        if req.level not in {lv.id for lv in s.levels}:
            raise HTTPException(422, "not a level on the scale")
        doc.classification, doc.classification_source = req.level, "manual"
    await db.commit()
    return {"classification": doc.classification, "source": doc.classification_source}
