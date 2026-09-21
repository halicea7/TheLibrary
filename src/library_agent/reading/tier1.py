"""Tier 1: structural reading.

Orientation card, one structured pass per section, document summary, categories. Roughly
5 calls for a paper and 30 for a book -- minutes, not the hours a per-chunk interpretive
pass would cost. Most documents stay at this tier permanently.

Every artifact records the model and prompt version that produced it, so bumping a prompt
regenerates only what that prompt owns."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy import delete, select, update
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
from library_agent.ingest.chunk import build_context_prefix
from library_agent.library import taxonomy
from library_agent.llm import providers
from library_agent.llm.client import LLM
from library_agent.llm.embed import embed_texts
from library_agent.llm.ollama import Ollama
from library_agent.reading import genre as genre_mod
from library_agent.reading import prompts

log = logging.getLogger(__name__)

# How much of the document feeds the orientation card.
OPENING_CHARS = 4000
CLOSING_CHARS = 2500
SECTION_CHARS = 6000


@dataclass
class Tier1Result:
    document_id: uuid.UUID
    title: str
    sections_read: int
    sections_skipped: int
    categories: list[str] = field(default_factory=list)
    orientation: str = ""
    figures_described: int = 0
    reembedded_chunks: int = 0


def _version(kind: str) -> str:
    return settings().prompt_versions.get(kind, "v1")


async def _upsert_artifact(
    db: AsyncSession,
    *,
    kind: str,
    target_kind: str,
    target_id: uuid.UUID,
    text: str,
    data: dict | None,
    model: str,
    prompt_version: str,
    tier: int = 1,
) -> Artifact:
    existing = (
        await db.execute(
            select(Artifact).where(
                Artifact.kind == kind,
                Artifact.target_kind == target_kind,
                Artifact.target_id == target_id,
            )
        )
    ).scalar_one_or_none()
    if existing:
        # Replacing the text invalidates the old vector; drop it so it is rewritten.
        await db.execute(
            delete(Embedding).where(
                Embedding.owner_kind == OwnerKind.ARTIFACT, Embedding.owner_id == existing.id
            )
        )
        existing.text = text
        existing.data = data
        existing.model = model
        existing.prompt_version = prompt_version
        existing.tier = tier
        await db.flush()
        return existing
    row = Artifact(
        kind=kind,
        target_kind=target_kind,
        target_id=target_id,
        text=text,
        data=data,
        model=model,
        prompt_version=prompt_version,
        tier=tier,
    )
    db.add(row)
    await db.flush()
    return row


async def stale_documents(db: AsyncSession) -> list[uuid.UUID]:
    """Documents whose Tier 1 artifacts predate the current prompt versions or model."""
    rows = (
        await db.execute(
            select(Artifact.target_id).where(
                Artifact.kind == ArtifactKind.DOCUMENT_SUMMARY,
                (Artifact.prompt_version != _version("document_summary"))
                | (Artifact.model != providers.model_for("reading")),
            )
        )
    ).scalars()
    return list(rows)


async def _document_text(db: AsyncSession, document_id: uuid.UUID) -> str:
    """Reconstruct document text from its chunks; the original file is not re-parsed."""
    rows = (
        await db.execute(
            select(Chunk.text).where(Chunk.document_id == document_id).order_by(Chunk.order_index)
        )
    ).scalars()
    return "\n\n".join(rows)


async def run_tier1(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    client: Ollama | None = None,
    progress=None,
    gate=None,
    force: bool = False,
) -> Tier1Result:
    cfg = settings()
    model = providers.model_for("reading")

    doc = (await db.execute(select(Document).where(Document.id == document_id))).scalar_one()
    sections = list(
        (
            await db.execute(
                select(Section)
                .where(Section.document_id == document_id)
                .order_by(Section.order_index)
            )
        ).scalars()
    )
    if not sections:
        raise ValueError(f"document {document_id} has no sections; run Tier 0 first")

    own = client is None
    c = client or LLM()
    try:
        full_text = await _document_text(db, document_id)
        toc = "\n".join(
            f"{'  ' * (s.level - 1)}- {s.path or s.title or '(untitled)'}" for s in sections
        )

        # --- 1. orientation card -------------------------------------------------
        orient = await c.structured(
            model,
            prompts.ORIENTATION_PROMPT.format(
                title=doc.title,
                toc=toc[:4000],
                opening=full_text[:OPENING_CHARS],
                closing=full_text[-CLOSING_CHARS:] if len(full_text) > OPENING_CHARS else "",
            ),
            prompts.ORIENTATION_SCHEMA,
            system=prompts.SYSTEM_LIBRARIAN,
        )
        one_liner = (orient.get("one_liner") or "").strip()
        # The maker's word beats the model's guess: a cartridge marked "runbook" reads as
        # runbooks even where a page looks like a report.
        if not doc.genre:
            doc.genre = genre_mod.normalise(orient.get("document_kind"))
        genre = doc.genre or "other"
        await _upsert_artifact(
            db,
            kind=ArtifactKind.ORIENTATION,
            target_kind=TargetKind.DOCUMENT,
            target_id=doc.id,
            text=orient.get("about", ""),
            data=orient,
            model=model,
            prompt_version=_version("orientation"),
        )
        if progress:
            await progress(1, len(sections) + 3)

        existing_categories = await taxonomy.canonical_names(db)
        guidance = prompts.category_guidance(existing_categories)

        # --- 2. per-section structured pass ---------------------------------------
        summaries: list[tuple[Section, str]] = []
        skipped = 0
        previous = ""
        bodies: dict[uuid.UUID, str] = {}
        for s in sections:
            bodies[s.id] = "\n\n".join(
                (
                    await db.execute(
                        select(Chunk.text)
                        .where(Chunk.section_id == s.id)
                        .order_by(Chunk.order_index)
                    )
                ).scalars()
            )
        # A section too short to summarise on its own is skipped -- but a short note
        # whose every section is short must still get read. Fold it into one section.
        min_chars = 200
        read_sections = sections
        if sections and not any(len(b.strip()) >= min_chars for b in bodies.values()):
            whole = "\n\n".join(bodies[s.id] for s in sections).strip()
            if len(whole) >= 60:
                lead = sections[0]
                bodies = {lead.id: whole}
                read_sections = [lead]
                min_chars = 60

        for i, s in enumerate(read_sections):
            # Yield to an active chat session between sections, never mid-call.
            if gate:
                await gate()
            body = bodies.get(s.id, "")
            if len(body.strip()) < min_chars:
                skipped += 1
                continue
            try:
                out = await c.structured(
                    model,
                    prompts.SECTION_PROMPT.format(
                        title=doc.title,
                        orientation=one_liner,
                        previous=f"Previous section covered: {previous}\n" if previous else "",
                        section_path=s.path or s.title or "(untitled)",
                        text=body[:SECTION_CHARS],
                        claims_guidance=genre_mod.CLAIMS_BY_GENRE.get(
                            genre, genre_mod.CLAIMS_BY_GENRE["other"]
                        ),
                        category_guidance=guidance,
                    ),
                    prompts.SECTION_SCHEMA,
                    system=prompts.SYSTEM_LIBRARIAN,
                )
            except Exception:
                log.warning("section pass failed for %s", s.id, exc_info=True)
                skipped += 1
                continue

            summary = (out.get("summary") or "").strip()
            if not summary:
                skipped += 1
                continue
            await _upsert_artifact(
                db,
                kind=ArtifactKind.SECTION_SUMMARY,
                target_kind=TargetKind.SECTION,
                target_id=s.id,
                text=summary,
                data=out,
                model=model,
                prompt_version=_version("section_summary"),
            )
            summaries.append((s, summary))
            previous = summary[:300]

            if cats := [x for x in (out.get("categories") or []) if isinstance(x, str)]:
                chunk_ids = list(
                    (await db.execute(select(Chunk.id).where(Chunk.section_id == s.id))).scalars()
                )
                await taxonomy.assign_chunks(db, chunk_ids, cats[:3])
            if progress:
                await progress(i + 2, len(read_sections) + 3)

        if not summaries:
            raise ValueError(f"no section produced a summary for {doc.title!r}")

        # --- 3. document summary ---------------------------------------------------
        joined = "\n\n".join(f"[{s.path or s.title or '?'}] {txt}" for s, txt in summaries)
        doc_out = await c.structured(
            model,
            prompts.DOCUMENT_SUMMARY_PROMPT.format(
                title=doc.title,
                orientation=one_liner,
                sections=joined[:18000],
                category_guidance=guidance,
            ),
            prompts.DOCUMENT_SUMMARY_SCHEMA,
            system=prompts.SYSTEM_LIBRARIAN,
        )
        doc_summary = (doc_out.get("summary") or "").strip()
        doc_artifact = await _upsert_artifact(
            db,
            kind=ArtifactKind.DOCUMENT_SUMMARY,
            target_kind=TargetKind.DOCUMENT,
            target_id=doc.id,
            text=doc_summary,
            data=doc_out,
            model=model,
            prompt_version=_version("document_summary"),
        )

        # --- 4. categories ----------------------------------------------------------
        cats = await taxonomy.assign_document(
            db, doc.id, [x for x in (doc_out.get("categories") or []) if isinstance(x, str)][:4]
        )
        # and one place on the shelf, if the shelf has been organised yet
        try:
            from library_agent.library.shelving import place_document

            await place_document(db, doc.id, client=c)
        except Exception:
            log.warning("could not shelve %s", doc.id, exc_info=True)

        # --- 5. embed the new artifacts ---------------------------------------------
        # The document summary replaces the Tier 0 fingerprint as the router's vector.
        to_embed: list[tuple[uuid.UUID, str]] = [(doc_artifact.id, f"{doc.title}\n\n{doc_summary}")]
        for s, txt in summaries:
            art = (
                await db.execute(
                    select(Artifact).where(
                        Artifact.kind == ArtifactKind.SECTION_SUMMARY,
                        Artifact.target_id == s.id,
                    )
                )
            ).scalar_one()
            to_embed.append((art.id, f"{doc.title} › {s.path or ''}\n\n{txt}"))

        vectors = await embed_texts([t for _, t in to_embed], c)
        for (aid, _), vec in zip(to_embed, vectors, strict=True):
            db.add(
                Embedding(
                    owner_kind=OwnerKind.ARTIFACT, owner_id=aid, model=cfg.embed_model, vec=vec
                )
            )

        # --- 6. optionally refresh chunk context prefixes with the orientation --------
        # OFF by default: measured, and it does not work. Appending the document's
        # one-liner to every chunk of that document adds a constant component to all its
        # vectors, making intra-document chunks *less* distinguishable. Measured against
        # the Phase 2 baseline it was flat-to-negative everywhere (keyword MRR -0.035,
        # natural recall@1 -0.015) for the cost of re-embedding the whole corpus.
        #
        # Anthropic's contextual retrieval generates context *per chunk*; a
        # document-constant string is the degenerate case of that idea. Revisit as a
        # Tier 2 experiment with genuinely per-chunk context.
        reembedded = 0
        if one_liner and cfg.orientation_in_chunk_prefix:
            section_by_id = {s.id: s for s in sections}
            chunks = list(
                (
                    await db.execute(
                        select(Chunk).where(Chunk.document_id == doc.id).order_by(Chunk.order_index)
                    )
                ).scalars()
            )
            changed: list[Chunk] = []
            for ch in chunks:
                prefix = build_context_prefix(
                    doc.title, section_by_id.get(ch.section_id), one_liner
                )
                if prefix != ch.context_prefix:
                    ch.context_prefix = prefix
                    changed.append(ch)
            if changed:
                vecs = await embed_texts([f"{ch.context_prefix}\n\n{ch.text}" for ch in changed], c)
                for ch, vec in zip(changed, vecs, strict=True):
                    await db.execute(
                        delete(Embedding).where(
                            Embedding.owner_kind == OwnerKind.CHUNK, Embedding.owner_id == ch.id
                        )
                    )
                    db.add(
                        Embedding(
                            owner_kind=OwnerKind.CHUNK,
                            owner_id=ch.id,
                            model=cfg.embed_model,
                            vec=vec,
                        )
                    )
                reembedded = len(changed)

        await db.execute(update(Document).where(Document.id == doc.id).values(tier=1))
        await db.flush()

        # The figures, read by the vision model into passages. Its own failure is not
        # the reading's: a paper without its figures described is still read.
        try:
            from library_agent.reading.figures import describe_figures

            fr = await describe_figures(db, doc.id, client=c, gate=gate)
            figures_described = fr.described
        except Exception:
            log.warning("figure pass failed for %s", doc.title[:40], exc_info=True)
            figures_described = 0

        return Tier1Result(
            document_id=doc.id,
            title=doc.title,
            sections_read=len(summaries),
            sections_skipped=skipped,
            categories=[c.name for c in cats],
            orientation=one_liner,
            reembedded_chunks=reembedded,
            figures_described=figures_described,
        )
    finally:
        if own:
            await c.aclose()
