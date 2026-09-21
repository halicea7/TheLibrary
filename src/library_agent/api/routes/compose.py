"""Composing documents: stream for the UI, then save, shelve, list."""

from __future__ import annotations

import json
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from library_agent.chat import compose as comp
from library_agent.db.session import SessionDep
from library_agent.ingest.tier0 import ingest
from library_agent.library.shelving import expand_category_ids
from library_agent.llm import providers

router = APIRouter(prefix="/api/compose", tags=["compose"])


class ComposeIn(BaseModel):
    brief: str = Field(min_length=4, max_length=4000)
    model: str | None = None
    length: str = "medium"
    category_ids: list[uuid.UUID] = Field(default_factory=list)
    cartridge_ids: list[uuid.UUID] = Field(default_factory=list)
    scope_label: str = ""


class DocIn(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    markdown: str = Field(min_length=1, max_length=2_000_000)


@router.post("")
async def compose_stream(req: ComposeIn, db: SessionDep) -> EventSourceResponse:
    if req.length not in comp.LENGTHS:
        raise HTTPException(422, f"length is one of {sorted(comp.LENGTHS)}")
    cats = await expand_category_ids(db, req.category_ids) if req.category_ids else None
    model = providers.chat_options().get(req.model or "", req.model) if req.model else None

    async def events() -> AsyncIterator[dict]:
        async for ev in comp.compose(
            req.brief,
            model=model,
            length=req.length,
            category_ids=cats,
            cartridge_ids=req.cartridge_ids or None,
            scope_label=req.scope_label,
        ):
            payload = ev["data"]
            yield {
                "event": ev["event"],
                "data": payload if isinstance(payload, str) else json.dumps(payload),
            }

    return EventSourceResponse(events())


@router.post("/save")
async def save(doc: DocIn) -> dict:
    path = comp.save_markdown(doc.title, doc.markdown)
    return {"file": path.name, "path": str(path).replace(str(Path.home()), "~")}


@router.get("/saved")
async def saved() -> list[dict]:
    return comp.list_saved()


@router.get("/saved/{name}")
async def download(name: str) -> FileResponse:
    if "/" in name or not name.endswith(".md"):
        raise HTTPException(404, "no such composition")
    path = comp.compositions_dir() / name
    if not path.exists():
        raise HTTPException(404, "no such composition")
    return FileResponse(path, media_type="text/markdown", filename=name)


@router.post("/shelve")
async def shelve(doc: DocIn, db: SessionDep) -> dict:
    """The document becomes a volume: saved, then ingested like any dropped file, so the
    library can read what it wrote."""
    path = comp.save_markdown(doc.title, doc.markdown)
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(doc.markdown)
        tmp_path = Path(tmp.name)
    try:
        r = await ingest(db, tmp_path, original_filename=f"{comp.slugify(doc.title)}.md")
        await db.commit()
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(422, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    return {
        "document_id": str(r.document_id),
        "title": r.title,
        "status": r.status,
        "file": path.name,
    }
