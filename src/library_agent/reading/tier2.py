"""Tier 2: deep interpretive reading.

This is the spec's original idea — the agent actually *reads*, producing reflections rather
than extraction. It is also the expensive tier, which is why it is demand-driven rather than
the default.

The spec called one LLM call per chunk. Here a whole section's chunks go in one structured
call: it amortises prefill across 5-15 passages and gives the model the neighbouring
passages as context, which is what makes a reflection able to say "this contradicts the
setup three paragraphs ago"."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import (
    Artifact,
    ArtifactKind,
    Chunk,
    Document,
    Embedding,
    OwnerKind,
    Section,
    TargetKind,
)
from library_agent.llm import providers
from library_agent.llm.client import LLM
from library_agent.llm.embed import embed_texts
from library_agent.llm.ollama import Ollama
from library_agent.reading import prompts
from library_agent.reading.tier1 import _upsert_artifact

log = logging.getLogger(__name__)

MAX_PASSAGE_CHARS = 1800


@dataclass
class Tier2Result:
    document_id: uuid.UUID
    title: str
    sections: int
    reflections: int
    skipped: int


async def _orientation(db: AsyncSession, document_id: uuid.UUID) -> str:
    row = (
        await db.execute(
            select(Artifact).where(
                Artifact.kind == ArtifactKind.ORIENTATION, Artifact.target_id == document_id
            )
        )
    ).scalar_one_or_none()
    if row and row.data:
        return str(row.data.get("one_liner") or "")
    return ""


async def run_tier2(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    client: Ollama | None = None,
    progress=None,
    gate=None,
) -> Tier2Result:
    cfg = settings()
    model = providers.load().models.get("reading") or cfg.deep_reader_model

    doc = (await db.execute(select(Document).where(Document.id == document_id))).scalar_one()
    if doc.tier < 1:
        raise ValueError(f"{doc.title!r} is tier {doc.tier}; run Tier 1 first")

    sections = list(
        (
            await db.execute(
                select(Section)
                .where(Section.document_id == document_id)
                .order_by(Section.order_index)
            )
        ).scalars()
    )
    orientation = await _orientation(db, document_id)

    own = client is None
    c = client or LLM()
    made = skipped = 0
    previous = ""
    try:
        for n, s in enumerate(sections):
            if gate:
                await gate()
            chunks = list(
                (
                    await db.execute(
                        select(Chunk).where(Chunk.section_id == s.id).order_by(Chunk.order_index)
                    )
                ).scalars()
            )
            if not chunks:
                continue

            passages = "\n\n".join(
                f"[{i}] {ch.text[:MAX_PASSAGE_CHARS]}" for i, ch in enumerate(chunks)
            )
            try:
                out = await c.structured(
                    model,
                    prompts.REFLECTION_PROMPT.format(
                        title=doc.title,
                        orientation=orientation,
                        section_path=s.path or s.title or "(untitled)",
                        previous=f"Previous section: {previous}\n" if previous else "",
                        passages=passages,
                    ),
                    prompts.REFLECTION_SCHEMA,
                    system=prompts.SYSTEM_LIBRARIAN,
                    temperature=0.5,  # interpretation needs more room than extraction
                )
            except Exception:
                log.warning("reflection pass failed for section %s", s.id, exc_info=True)
                skipped += len(chunks)
                continue

            by_index = {
                int(r["index"]): str(r.get("reflection") or "").strip()
                for r in (out.get("reflections") or [])
                if isinstance(r, dict) and "index" in r
            }
            to_embed: list[tuple[uuid.UUID, str]] = []
            for i, ch in enumerate(chunks):
                body = by_index.get(i, "")
                if len(body) < 40:
                    skipped += 1
                    continue
                art = await _upsert_artifact(
                    db,
                    kind=ArtifactKind.REFLECTION,
                    target_kind=TargetKind.CHUNK,
                    target_id=ch.id,
                    text=body,
                    data=None,
                    model=model,
                    prompt_version=cfg.prompt_versions.get("reflection", "v1"),
                    tier=2,
                )
                to_embed.append((art.id, f"{doc.title} › {s.path}\n\n{body}"))
                made += 1

            if to_embed:
                vecs = await embed_texts([t for _, t in to_embed], c)
                for (aid, _), vec in zip(to_embed, vecs, strict=True):
                    db.add(
                        Embedding(
                            owner_kind=OwnerKind.ARTIFACT,
                            owner_id=aid,
                            model=cfg.embed_model,
                            vec=vec,
                        )
                    )
            previous = (s.path or s.title or "")[:120]
            if progress:
                await progress(n + 1, len(sections))

        await db.execute(update(Document).where(Document.id == doc.id).values(tier=2))
        await db.flush()
    finally:
        if own:
            await c.aclose()

    return Tier2Result(
        document_id=doc.id,
        title=doc.title,
        sections=len(sections),
        reflections=made,
        skipped=skipped,
    )


async def promotion_candidates(
    db: AsyncSession, limit: int = 10
) -> list[tuple[uuid.UUID, str, str]]:
    """Documents that have earned a deep read: starred, frequently retrieved, or cited by
    many others in the corpus. Demand-driven rather than a chore."""
    rows = (
        await db.execute(
            text("""
            select d.id, d.title,
                   case when d.starred then 'starred'
                        when cites.n >= 2 then 'cited by ' || cites.n || ' in corpus'
                        else 'retrieved ' || d.retrieval_hits || 'x' end as reason,
                   (d.starred::int * 1000) + (coalesce(cites.n,0) * 50) + d.retrieval_hits as score
            from document d
            left join (
                select matched_document_id as id, count(*) as n
                from citation where matched_document_id is not null
                group by matched_document_id
            ) cites on cites.id = d.id
            where d.tier = 1
            order by score desc, d.added_at
            limit :lim
            """),
            {"lim": limit},
        )
    ).all()
    return [(r.id, r.title, r.reason) for r in rows]
