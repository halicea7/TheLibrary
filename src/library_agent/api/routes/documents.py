"""Document upload, listing, and deletion."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.api.schemas import DocumentDetail, DocumentOut, SectionOut, UploadResult
from library_agent.config import settings
from library_agent.db.models import (
    Artifact,
    ArtifactKind,
    Cartridge,
    Category,
    Chunk,
    Document,
    DocumentCategory,
    Section,
)
from library_agent.db.purge import delete_document
from library_agent.db.session import SessionDep
from library_agent.ingest import figures
from library_agent.ingest.dedup import find_exact
from library_agent.ingest.extract import SUPPORTED, content_hash, extract
from library_agent.ingest.tier0 import _store as store_file
from library_agent.ingest.tier0 import ingest
from library_agent.library.cartridge import cartridge_provenance
from library_agent.llm import providers
from library_agent.worker.tasks import enqueue_ocr

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


async def _counts(db: AsyncSession, doc_ids: list[uuid.UUID]) -> dict[uuid.UUID, tuple[int, int]]:
    if not doc_ids:
        return {}
    sec = dict(
        (
            await db.execute(
                select(Section.document_id, func.count())
                .where(Section.document_id.in_(doc_ids))
                .group_by(Section.document_id)
            )
        ).all()
    )
    ch = dict(
        (
            await db.execute(
                select(Chunk.document_id, func.count())
                .where(Chunk.document_id.in_(doc_ids))
                .group_by(Chunk.document_id)
            )
        ).all()
    )
    return {d: (sec.get(d, 0), ch.get(d, 0)) for d in doc_ids}


def _to_out(d: Document, sections: int, chunks: int) -> DocumentOut:
    return DocumentOut(
        id=d.id,
        title=d.title,
        kind=d.kind,
        status=d.status,
        tier=d.tier,
        page_count=d.page_count,
        original_filename=d.original_filename,
        sections=sections,
        chunks=chunks,
        starred=d.starred,
        near_dup_of=d.near_dup_of,
        added_at=d.added_at,
        readings_only=d.readings_only,
        genre=d.genre,
    )


@router.get("", response_model=list[DocumentOut])
async def list_documents(db: SessionDep) -> list[DocumentOut]:
    docs = list((await db.execute(select(Document).order_by(Document.added_at.desc()))).scalars())
    counts = await _counts(db, [d.id for d in docs])
    cats: dict[uuid.UUID, list[str]] = {}
    for did, name in (
        await db.execute(
            select(DocumentCategory.document_id, Category.name)
            .join(Category, Category.id == DocumentCategory.category_id)
            .order_by(Category.name)
        )
    ).all():
        cats.setdefault(did, []).append(name)
    provenance = await cartridge_provenance(db, [d.id for d in docs])
    shelves = await shelf_labels(db)
    out = []
    for d in docs:
        o = _to_out(d, *counts.get(d.id, (0, 0)))
        o.categories = cats.get(d.id, [])
        o.cartridge = provenance.get(d.id)
        o.shelf = shelves.get(d.shelf_id) if d.shelf_id else None
        out.append(o)
    return out


async def shelf_labels(db: AsyncSession) -> dict[uuid.UUID, dict]:
    """sub-shelf id -> {top, top_id, sub, sub_id}."""
    rows = list((await db.execute(select(Category))).scalars())
    by_id = {c.id: c for c in rows}
    out = {}
    for c in rows:
        if c.parent_id and c.parent_id in by_id:
            p = by_id[c.parent_id]
            out[c.id] = {"top": p.name, "top_id": str(p.id), "sub": c.name, "sub_id": str(c.id)}
    return out


@router.post("", response_model=UploadResult)
async def upload_document(
    file: Annotated[UploadFile, File()],
    db: SessionDep,
    force: bool = False,
) -> UploadResult:
    name = file.filename or "upload"
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(
            415, f"unsupported type {Path(name).suffix!r}; supported: {sorted(SUPPORTED)}"
        )

    started = time.time()
    # Tier 0 is seconds and has no LLM call, so it runs inline. Tier 1 is the part that
    # goes to the job queue.
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        # A scanned PDF has no text layer; rather than refuse it, transcribe it with the
        # vision model in the background. The upload returns at once, the document appears
        # on the shelf when the transcription finishes.
        ex = None
        try:
            ex = extract(tmp_path)
        except Exception:  # noqa: BLE001 -- let ingest() raise the precise error below
            ex = None
        cfg = settings()
        if (
            ex is not None
            and ex.needs_ocr
            and Path(name).suffix.lower() == ".pdf"
            and cfg.ocr_enabled
            and providers.model_for("vision")
        ):
            chash = content_hash(tmp_path)
            if not force and (existing := await find_exact(db, chash)):
                return UploadResult(
                    document_id=existing.id,
                    title=existing.title,
                    status="duplicate",
                    sections=0,
                    chunks=0,
                    pages=existing.page_count or 0,
                    duplicate_of=existing.id,
                    elapsed_seconds=round(time.time() - started, 2),
                )
            stored = store_file(tmp_path, chash)
            redis = await create_pool(RedisSettings.from_dsn(cfg.redis_url))
            try:
                await enqueue_ocr(redis, str(stored), name)
            finally:
                await redis.aclose()
            return UploadResult(
                document_id=None,
                title=Path(name).stem,
                status="ocr",
                sections=0,
                chunks=0,
                pages=ex.page_count,
                needs_ocr=True,
                elapsed_seconds=round(time.time() - started, 2),
            )
        result = await ingest(db, tmp_path, original_filename=name, force=force, extracted=ex)
        await db.commit()
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(422, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadResult(
        document_id=result.document_id,
        title=result.title,
        status=result.status,
        sections=result.sections,
        chunks=result.chunks,
        pages=result.pages,
        duplicate_of=result.duplicate_of,
        near_duplicate_sim=result.near_duplicate_sim,
        needs_ocr=result.needs_ocr,
        elapsed_seconds=round(time.time() - started, 2),
    )


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: uuid.UUID, db: SessionDep) -> DocumentDetail:
    doc = (
        await db.execute(select(Document).where(Document.id == document_id))
    ).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "document not found")

    sections = list(
        (
            await db.execute(
                select(Section)
                .where(Section.document_id == document_id)
                .order_by(Section.order_index)
            )
        ).scalars()
    )
    per_section = dict(
        (
            await db.execute(
                select(Chunk.section_id, func.count())
                .where(Chunk.document_id == document_id)
                .group_by(Chunk.section_id)
            )
        ).all()
    )
    counts = await _counts(db, [document_id])
    base = _to_out(doc, *counts.get(document_id, (0, 0)))
    base.cartridge = (await cartridge_provenance(db, [document_id])).get(document_id)
    base.shelf = (await shelf_labels(db)).get(doc.shelf_id) if doc.shelf_id else None
    base.categories = list(
        (
            await db.execute(
                select(Category.name)
                .join(DocumentCategory, DocumentCategory.category_id == Category.id)
                .where(DocumentCategory.document_id == document_id)
                .order_by(Category.name)
            )
        ).scalars()
    )
    return DocumentDetail(
        **base.model_dump(),
        sections_detail=[
            SectionOut(
                id=s.id,
                title=s.title,
                path=s.path,
                level=s.level,
                page_start=s.page_start,
                page_end=s.page_end,
                chunks=per_section.get(s.id, 0),
            )
            for s in sections
        ],
    )


@router.delete("/{document_id}")
async def remove_document(document_id: uuid.UUID, db: SessionDep) -> dict[str, int | str]:
    doc = (
        await db.execute(select(Document).where(Document.id == document_id))
    ).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "document not found")
    vectors, artifacts = await delete_document(db, document_id)
    await db.commit()
    return {"deleted": str(document_id), "vectors_removed": vectors, "artifacts_removed": artifacts}


@router.get("/{document_id}/reflections")
async def document_reflections(
    document_id: uuid.UUID, db: SessionDep, limit: int = 100
) -> list[dict]:
    """What the agent thought while reading. Exposing this is the only practical way to
    judge whether Tier 2 output is worth its hours."""
    rows = (
        await db.execute(
            select(Chunk, Artifact, Section)
            .join(
                Artifact,
                (Artifact.target_id == Chunk.id) & (Artifact.kind == ArtifactKind.REFLECTION),
            )
            .outerjoin(Section, Section.id == Chunk.section_id)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.order_index)
            .limit(limit)
        )
    ).all()
    return [
        {
            "chunk_id": str(ch.id),
            "section": sec.path if sec else "",
            "page": ch.page_start,
            "passage": ch.text[:600],
            "reflection": art.text,
            "model": art.model,
            "prompt_version": art.prompt_version,
        }
        for ch, art, sec in rows
    ]


def _deoverlap(texts: list[str], *, min_words: int = 8) -> list[str]:
    """Chunks are windowed with ~12% overlap so retrieval never loses a sentence at a
    boundary. Read linearly, that overlap is every paragraph's tail printed twice.

    Matched on words, not characters: the chunker whitespace-normalises the tail it
    carries forward, so the two copies differ in line breaks and never match verbatim."""
    import re

    out: list[str] = []
    for i, cur in enumerate(texts):
        if i == 0:
            out.append(cur)
            continue
        prev_w = texts[i - 1].split()
        cur_w = cur.split()
        drop = 0
        for k in range(min(len(prev_w), len(cur_w), 400), min_words - 1, -1):
            if prev_w[-k:] == cur_w[:k]:
                drop = k
                break
        if not drop:
            out.append(cur)
            continue
        # Skip `drop` words in the original text, preserving whatever whitespace follows.
        m = list(re.finditer(r"\S+", cur))
        cut = m[drop - 1].end() if drop <= len(m) else len(cur)
        out.append(cur[cut:].lstrip())
    return out


def _real_heading(title: str | None) -> bool:
    """Figure labels and table headers ('BERT', 'E1 E2', 'Bert Bert') get detected as
    headings. A heading worth showing has a word of five or more letters."""
    if not title:
        return False
    import re

    return bool(re.search(r"[A-Za-z]{5,}", title))


@router.get("/{document_id}/read")
async def read_document(document_id: uuid.UUID, db: SessionDep) -> dict:
    """The document as a reader sees it: sections in order, each with its passages, plus
    whatever the library has written about them — section summaries at Tier 1 and
    per-passage reflections at Tier 2, ready to sit in the margin beside their text."""
    doc = (
        await db.execute(select(Document).where(Document.id == document_id))
    ).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "document not found")

    sections = list(
        (
            await db.execute(
                select(Section)
                .where(Section.document_id == document_id)
                .order_by(Section.order_index)
            )
        ).scalars()
    )
    chunks = list(
        (
            await db.execute(
                select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.order_index)
            )
        ).scalars()
    )
    arts = list(
        (
            await db.execute(
                select(Artifact).where(
                    Artifact.kind.in_([ArtifactKind.SECTION_SUMMARY, ArtifactKind.REFLECTION]),
                    Artifact.target_id.in_([s.id for s in sections] + [c.id for c in chunks]),
                )
            )
        ).scalars()
    )
    # Marginalia has an author: yours carries no cartridge; theirs is labelled with it.
    cart_ids = {a.cartridge_id for a in arts if a.cartridge_id}
    carts = (
        {
            c.id: {"id": str(c.id), "name": c.name, "colour": c.colour}
            for c in (
                await db.execute(select(Cartridge).where(Cartridge.id.in_(cart_ids)))
            ).scalars()
        }
        if cart_ids
        else {}
    )
    summaries = {a.target_id: a.text for a in arts if a.kind == ArtifactKind.SECTION_SUMMARY}
    if doc.readings_only:
        # The body *is* the reading; announcing it again above would double it.
        summaries = {}
    reflections: dict[uuid.UUID, list[dict]] = {}
    for a in sorted(
        (a for a in arts if a.kind == ArtifactKind.REFLECTION),
        key=lambda a: (a.cartridge_id is not None, a.created_at),
    ):
        reflections.setdefault(a.target_id, []).append(
            {"text": a.text, "cartridge": carts.get(a.cartridge_id) if a.cartridge_id else None}
        )

    # Figure passages are the vision model's readings; they belong under the figure,
    # not in the running text.
    figure_text = {}
    by_section: dict[uuid.UUID | None, list[Chunk]] = {}
    for c in chunks:
        if c.kind == "figure":
            m = re.match(r"Figure (\d+),", c.text)
            if m:
                figure_text[int(m.group(1))] = c.text.split("\n\n", 1)[-1]
            continue
        by_section.setdefault(c.section_id, []).append(c)

    out_sections = []
    for s in sections:
        cs = by_section.get(s.id, [])
        if doc.readings_only and not cs:
            # Their library had nothing to say about this section; a bare heading
            # would only advertise an absence.
            continue
        texts = _deoverlap([c.text for c in cs])
        # The first chunk of a section usually opens with the heading line itself, which
        # the reader already shows as the heading. Drop it once.
        if texts and s.title:
            first_line, _, rest = texts[0].partition("\n")
            # In a markdown volume the heading line still wears its hashes.
            if first_line.strip().lstrip("#").strip().lower() == s.title.strip().lower():
                texts[0] = rest.lstrip()
        out_sections.append(
            {
                "id": str(s.id),
                # A junk heading still keeps its passages; it just isn't announced.
                "title": s.title if _real_heading(s.title) else None,
                "path": s.path,
                "level": s.level,
                "page_start": s.page_start,
                "summary": summaries.get(s.id),
                "passages": [
                    {
                        "id": str(c.id),
                        "page": c.page_start,
                        "text": txt,
                        "reflection": (reflections.get(c.id) or [{}])[0].get("text"),
                        "reflections": reflections.get(c.id, []),
                    }
                    for c, txt in zip(cs, texts, strict=True)
                ],
            }
        )
    if orphans := by_section.get(None):
        out_sections.append(
            {
                "id": None,
                "title": None,
                "path": "",
                "level": 1,
                "page_start": None,
                "summary": None,
                "passages": [
                    {
                        "id": str(c.id),
                        "page": c.page_start,
                        "text": c.text,
                        "reflection": (reflections.get(c.id) or [{}])[0].get("text"),
                        "reflections": reflections.get(c.id, []),
                    }
                    for c in orphans
                ],
            }
        )
    return {
        "id": str(doc.id),
        "title": doc.title,
        "tier": doc.tier,
        "kind": doc.kind,
        "readings_only": doc.readings_only,
        "genre": doc.genre,
        "cartridge": (await cartridge_provenance(db, [doc.id])).get(doc.id),
        "page_count": doc.page_count,
        "sections": out_sections,
        # Figures are read from the original, so a readings-only volume has none.
        "figures": [{**asdict(f), "description": figure_text.get(f.n)} for f in _figures(doc)],
    }


def _figures(doc: Document) -> list:
    if doc.readings_only or not doc.source_path:
        return []
    try:
        return figures.index(doc.content_hash, Path(doc.source_path))
    except Exception:
        log.debug("figure index failed for %s", doc.id, exc_info=True)
        return []


@router.get("/{document_id}/figures/{n}.png")
async def figure_png(document_id: uuid.UUID, n: int, db: SessionDep) -> FileResponse:
    """One figure, rendered from the original at 2x on first request and cached."""
    doc = (
        await db.execute(select(Document).where(Document.id == document_id))
    ).scalar_one_or_none()
    if not doc or doc.readings_only or not doc.source_path:
        raise HTTPException(404, "no such figure")
    path = await asyncio.to_thread(figures.render, doc.content_hash, Path(doc.source_path), n)
    if not path:
        raise HTTPException(404, "no such figure")
    return FileResponse(
        path, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"}
    )
