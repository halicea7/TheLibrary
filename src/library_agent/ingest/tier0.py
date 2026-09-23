"""Tier 0: extract, structure, chunk, embed. No LLM calls.

A document is searchable the moment this finishes -- seconds, not hours. Tier 1 and 2
enrich it later. That is what turns a 500-document backfill from a multi-week blocker
into something you can use the same afternoon."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import (
    Chunk as ChunkRow,
)
from library_agent.db.models import (
    Document,
    DocumentKind,
    DocumentStatus,
    Embedding,
    OwnerKind,
    chunk_id,
)
from library_agent.db.models import (
    Section as SectionRow,
)
from library_agent.db.purge import delete_document
from library_agent.ingest import dedup
from library_agent.ingest.chunk import chunk_document
from library_agent.ingest.extract import content_hash, extract
from library_agent.ingest.structure import build_sections
from library_agent.llm.client import LLM
from library_agent.llm.embed import embed_texts
from library_agent.llm.ollama import Ollama


@dataclass
class IngestResult:
    document_id: uuid.UUID
    title: str
    status: str
    sections: int
    chunks: int
    pages: int
    duplicate_of: uuid.UUID | None = None
    near_duplicate_sim: float | None = None
    needs_ocr: bool = False


def _guess_kind(path: Path, pages: int) -> str:
    if path.suffix.lower() in {".md", ".markdown", ".rst", ".html", ".htm"}:
        return DocumentKind.DOC
    if path.suffix.lower() in {".txt", ".text"}:
        return DocumentKind.NOTE
    return DocumentKind.BOOK if pages > 60 else DocumentKind.PAPER


def _store(src: Path, chash: str) -> Path:
    """Content-addressed storage so re-uploads never duplicate bytes on disk."""
    dest_dir = settings().storage_dir / chash[:2]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{chash}{src.suffix.lower()}"
    if not dest.exists():
        shutil.copy2(src, dest)
    return dest


async def ingest(
    db: AsyncSession,
    path: Path,
    *,
    original_filename: str | None = None,
    client: Ollama | None = None,
    force: bool = False,
) -> IngestResult:
    chash = content_hash(path)
    filename = original_filename or path.name

    if not force and (existing := await dedup.find_exact(db, chash)):
        return IngestResult(
            document_id=existing.id,
            title=existing.title,
            status="duplicate",
            sections=0,
            chunks=0,
            pages=existing.page_count or 0,
            duplicate_of=existing.id,
        )

    ex = extract(path)
    title = ex.title or Path(filename).stem
    sections = build_sections(ex)
    chunks = chunk_document(ex, sections, title)
    if not chunks:
        if ex.needs_ocr:
            raise ValueError(
                f"{filename} has almost no text layer — it looks like a scanned or "
                "image-only PDF. The library reads text, not page images; run it through "
                "OCR first (e.g. `ocrmypdf in.pdf out.pdf`) and add the result."
            )
        raise ValueError(f"no extractable text in {filename}")

    own_client = client is None
    c = client or LLM()
    try:
        # Prefix is embedded with the chunk: that is the whole point of contextual retrieval.
        vectors = await embed_texts([f"{k.context_prefix}\n\n{k.text}" for k in chunks], c)
        fp_vec = (await embed_texts([dedup.fingerprint_text(title, ex.text)], c))[0]
    finally:
        if own_client:
            await c.aclose()

    stored = _store(path, chash)

    # force=True re-ingests in place. Remove the old version first -- including its
    # polymorphic vectors/artifacts, which no cascade covers -- so the near-duplicate
    # lookup below cannot match this document's own stale fingerprint.
    if previous := await dedup.find_exact(db, chash):
        await delete_document(db, previous.id)
        await db.flush()

    near = await dedup.find_near_duplicate(db, fp_vec)

    doc = Document(
        content_hash=chash,
        title=title,
        source_path=str(stored),
        original_filename=filename,
        kind=_guess_kind(path, ex.page_count),
        page_count=ex.page_count,
        tier=0,
        status=DocumentStatus.READY,
        near_dup_of=near[0] if near else None,
    )
    db.add(doc)
    await db.flush()

    section_rows: dict[int, SectionRow] = {}
    for s in sections:
        row = SectionRow(
            document_id=doc.id,
            order_index=s.order_index,
            title=s.title,
            level=s.level,
            path=s.path,
            char_start=s.char_start,
            char_end=s.char_end,
            page_start=s.page_start,
            page_end=s.page_end,
        )
        db.add(row)
        section_rows[s.order_index] = row
    await db.flush()  # one round trip; every row now has an id

    for s in sections:
        if s.parent_index is not None and (parent := section_rows.get(s.parent_index)):
            section_rows[s.order_index].parent_id = parent.id
    section_ids = {i: r.id for i, r in section_rows.items()}

    model = settings().embed_model
    db.add(Embedding(owner_kind=OwnerKind.DOCUMENT, owner_id=doc.id, model=model, vec=fp_vec))

    seen: set[uuid.UUID] = set()
    for k, vec in zip(chunks, vectors, strict=True):
        cid = chunk_id(chash, k.char_start, k.char_end)
        if cid in seen:  # identical span produced twice: keep the first
            continue
        seen.add(cid)
        db.add(
            ChunkRow(
                id=cid,
                document_id=doc.id,
                section_id=section_ids.get(k.section_index)
                if k.section_index is not None
                else None,
                order_index=k.order_index,
                text=k.text,
                context_prefix=k.context_prefix,
                token_count=k.token_count,
                page_start=k.page_start,
                char_start=k.char_start,
                char_end=k.char_end,
            )
        )
        db.add(Embedding(owner_kind=OwnerKind.CHUNK, owner_id=cid, model=model, vec=vec))

    await db.flush()
    return IngestResult(
        document_id=doc.id,
        title=title,
        status=DocumentStatus.READY,
        sections=len(sections),
        chunks=len(seen),
        pages=ex.page_count,
        near_duplicate_sim=near[1] if near else None,
        duplicate_of=near[0] if near else None,
        needs_ocr=ex.needs_ocr,
    )
