"""Shared fixtures."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from sqlalchemy import select

from library_agent.config import settings
from library_agent.db.models import (
    Artifact,
    ArtifactKind,
    Category,
    Chunk,
    Document,
    DocumentCategory,
    DocumentStatus,
    Embedding,
    OwnerKind,
    Section,
    TargetKind,
    chunk_id,
)


@pytest.fixture
async def db():
    """A session whose work is thrown away."""
    try:
        from library_agent.db.session import engine
    except Exception:  # noqa: BLE001
        pytest.skip("no database")
    from sqlalchemy.ext.asyncio import AsyncSession

    async with engine().connect() as conn:
        tx = await conn.begin()
        s = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            yield s
        finally:
            await s.close()
            await tx.rollback()


BODY_A = "Alpha is a test document about cartridges. " * 8
BODY_B = "Beta is a second document that disagrees with alpha. " * 8


def _vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(settings().embed_dim).astype(np.float32).tolist()


async def make_document(
    db, *, title: str, body: str, tier: int = 2, subject: str = "Testing Shelf"
):
    """A read-and-annotated document with two sections, one chunk each, and one reflection."""
    h = uuid.uuid5(uuid.NAMESPACE_URL, body).hex + uuid.uuid5(uuid.NAMESPACE_DNS, body).hex
    h = h[:64]
    doc = Document(
        content_hash=h,
        title=title,
        source_path="",
        original_filename=f"{title}.md",
        tier=tier,
        status=DocumentStatus.READY,
    )
    db.add(doc)
    await db.flush()
    db.add(
        Embedding(
            owner_kind=OwnerKind.DOCUMENT,
            owner_id=doc.id,
            model=settings().embed_model,
            vec=_vec(1),
        )
    )
    halves = [body[: len(body) // 2], body[len(body) // 2 :]]
    for i, part in enumerate(halves):
        sec = Section(
            document_id=doc.id, order_index=i, path=f"Part {i}", title=f"Part {i}", page_start=i + 1
        )
        db.add(sec)
        await db.flush()
        ch = Chunk(
            id=chunk_id(h, i * 100, i * 100 + len(part)),
            document_id=doc.id,
            section_id=sec.id,
            order_index=i,
            text=part,
            page_start=i + 1,
        )
        db.add(ch)
        await db.flush()
        db.add(
            Embedding(
                owner_kind=OwnerKind.CHUNK,
                owner_id=ch.id,
                model=settings().embed_model,
                vec=_vec(10 + i),
            )
        )
        summ = Artifact(
            kind=ArtifactKind.SECTION_SUMMARY,
            target_kind=TargetKind.SECTION,
            target_id=sec.id,
            text=f"Reading of part {i}, in the summariser's own words.",
            model="test",
            prompt_version="v1",
        )
        db.add(summ)
        await db.flush()
        db.add(
            Embedding(
                owner_kind=OwnerKind.ARTIFACT,
                owner_id=summ.id,
                model=settings().embed_model,
                vec=_vec(20 + i),
            )
        )
        if tier >= 2:
            refl = Artifact(
                kind=ArtifactKind.REFLECTION,
                target_kind=TargetKind.CHUNK,
                target_id=ch.id,
                text=f"A note on part {i}",
                model="test",
                prompt_version="v1",
                tier=2,
            )
            db.add(refl)
            await db.flush()
            db.add(
                Embedding(
                    owner_kind=OwnerKind.ARTIFACT,
                    owner_id=refl.id,
                    model=settings().embed_model,
                    vec=_vec(30 + i),
                )
            )
    ds = Artifact(
        kind=ArtifactKind.DOCUMENT_SUMMARY,
        target_kind=TargetKind.DOCUMENT,
        target_id=doc.id,
        text=f"{title}, in short.",
        model="test",
        prompt_version="v1",
    )
    db.add(ds)
    await db.flush()
    db.add(
        Embedding(
            owner_kind=OwnerKind.ARTIFACT,
            owner_id=ds.id,
            model=settings().embed_model,
            vec=_vec(40),
        )
    )
    cat = (await db.execute(select(Category).where(Category.name == subject))).scalar_one_or_none()
    if not cat:
        cat = Category(name=subject)
        db.add(cat)
        await db.flush()
    db.add(DocumentCategory(document_id=doc.id, category_id=cat.id))
    await db.flush()
    return doc
