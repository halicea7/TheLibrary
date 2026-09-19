"""Document upload, listing, and deletion."""

from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.api.schemas import DocumentDetail, DocumentOut, SectionOut, UploadResult
from library_agent.db.models import (
    Artifact,
    ArtifactKind,
    Category,
    Chunk,
    Document,
    DocumentCategory,
    Section,
)
from library_agent.db.purge import delete_document
from library_agent.db.session import SessionDep
from library_agent.ingest.extract import SUPPORTED
from library_agent.ingest.tier0 import ingest

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
    out = []
    for d in docs:
        o = _to_out(d, *counts.get(d.id, (0, 0)))
        o.categories = cats.get(d.id, [])
        out.append(o)
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
        result = await ingest(db, tmp_path, original_filename=name, force=force)
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
    summaries = {a.target_id: a.text for a in arts if a.kind == ArtifactKind.SECTION_SUMMARY}
    reflections = {a.target_id: a.text for a in arts if a.kind == ArtifactKind.REFLECTION}

    by_section: dict[uuid.UUID | None, list[Chunk]] = {}
    for c in chunks:
        by_section.setdefault(c.section_id, []).append(c)

    out_sections = []
    for s in sections:
        cs = by_section.get(s.id, [])
        texts = _deoverlap([c.text for c in cs])
        # The first chunk of a section usually opens with the heading line itself, which
        # the reader already shows as the heading. Drop it once.
        if texts and s.title:
            first_line, _, rest = texts[0].partition("\n")
            if first_line.strip().lower() == s.title.strip().lower():
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
                        "reflection": reflections.get(c.id),
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
                        "reflection": reflections.get(c.id),
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
        "page_count": doc.page_count,
        "sections": out_sections,
    }
