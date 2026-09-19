"""Cartridges: list, preview, export, import, eject."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from library_agent.config import settings
from library_agent.db.session import SessionDep
from library_agent.library import cartridge as cart
from library_agent.worker.tasks import schedule_library_rebuild

router = APIRouter(prefix="/api/cartridges", tags=["cartridges"])


class Selection(BaseModel):
    document_ids: list[uuid.UUID] = Field(default_factory=list)
    category_ids: list[uuid.UUID] = Field(default_factory=list)
    cartridge_ids: list[uuid.UUID] = Field(default_factory=list)
    level: str = "readings"


class ExportRequest(Selection):
    name: str = Field(min_length=1, max_length=80)
    colour: str | None = None
    icon_svg: str | None = Field(default=None, max_length=20_000)
    made_by: str | None = Field(default=None, max_length=120)


@router.get("")
async def list_cartridges(db: SessionDep) -> dict:
    return {"cartridges": await cart.list_cartridges(db), "palette": cart.PALETTE}


@router.post("/preview")
async def preview(sel: Selection, db: SessionDep) -> dict:
    """What would leave, before it does. Exporting is a deliberate act."""
    try:
        ids = await cart.resolve_selection(
            db,
            document_ids=sel.document_ids,
            category_ids=sel.category_ids,
            cartridge_ids=sel.cartridge_ids,
        )
        bundle = await cart.gather(db, ids, sel.level)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "level": sel.level,
        "counts": bundle.counts(),
        "estimated_bytes": bundle.estimated_bytes(),
        "titles": [d["title"] for d in bundle.documents][:40],
    }


@router.post("/export")
async def export(req: ExportRequest, db: SessionDep) -> FileResponse:
    ids = await cart.resolve_selection(
        db,
        document_ids=req.document_ids,
        category_ids=req.category_ids,
        cartridge_ids=req.cartridge_ids,
    )
    if not ids:
        raise HTTPException(422, "nothing selected")
    try:
        path = await cart.build_cartridge(
            db,
            document_ids=ids,
            level=req.level,
            name=req.name,
            colour=req.colour,
            icon_svg=req.icon_svg,
            made_by=req.made_by,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return FileResponse(path, media_type="application/zip", filename=path.name)


@router.post("/import")
async def import_(file: Annotated[UploadFile, File()], db: SessionDep) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        if not cart.is_cartridge(tmp_path):
            raise HTTPException(415, "not a cartridge (no cartridge.json in the zip)")
        try:
            res = await cart.import_cartridge(db, tmp_path)
            await db.commit()
        except cart.CartridgeError as exc:
            await db.rollback()
            raise HTTPException(422, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    if not res.noop:
        # Threads and disagreements must now be recomputed across the new whole.
        redis = await create_pool(RedisSettings.from_dsn(settings().redis_url))
        try:
            await schedule_library_rebuild(redis)
        finally:
            await redis.aclose()
    return {
        "cartridge_id": str(res.cartridge_id),
        "name": res.name,
        "version": res.version,
        "noop": res.noop,
        "documents_introduced": res.documents_introduced,
        "documents_joined": res.documents_joined,
        "artifacts": res.artifacts,
        "artifacts_skipped": res.artifacts_skipped,
        "vectors_loaded": res.vectors_loaded,
        "vectors_embedded": res.vectors_embedded,
        "replaced_version": res.replaced_version,
    }


@router.delete("/{cartridge_id}")
async def eject(cartridge_id: uuid.UUID, db: SessionDep) -> dict:
    try:
        res = await cart.eject_cartridge(db, cartridge_id)
        await db.commit()
    except cart.CartridgeError as exc:
        raise HTTPException(404, str(exc)) from exc
    redis = await create_pool(RedisSettings.from_dsn(settings().redis_url))
    try:
        await schedule_library_rebuild(redis)
    finally:
        await redis.aclose()
    return {
        "documents_removed": res.documents_removed,
        "documents_kept": res.documents_kept,
        "artifacts_removed": res.artifacts_removed,
    }
