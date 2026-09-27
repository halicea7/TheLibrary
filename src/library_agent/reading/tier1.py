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
SUMMARY_CHARS = 18000  # section summaries the document entry reads at once


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


async def _digest_stretches(c, model, title, one_liner, summaries, gate=None) -> str:
    groups: list[list[tuple[Section, str]]] = [[]]
    size = 0
    for s, txt in summaries:
        line = len(txt) + 40
        if groups[-1] and size + line > SUMMARY_CHARS * 5 // 6:
            groups.append([])
            size = 0
        groups[-1].append((s, txt))
        size += line
    out = []
    for g in groups:
        if gate:
            await gate()
        first, last = g[0][0], g[-1][0]
        span = first.path or first.title or "?"
        if last is not first:
            span += " … " + (last.path or last.title or "?")
        digest = await c.generate(
            model,
            prompts.PART_SUMMARY_PROMPT.format(
                title=title,
                orientation=one_liner,
                span=span,
                sections="\n\n".join(f"[{s.path or s.title or '?'}] {t}" for s, t in g),
            ),
            system=prompts.SYSTEM_LIBRARIAN,
            num_predict=700,
        )
        out.append(f"[{span}] {digest.strip()}")
    return "\n\n".join(out)


def section_parts(pieces: list[str], limit: int) -> list[str]:
    """A section's passages grouped, in order, into parts of at most `limit` characters --
    split between passages, never inside one (a passage longer than the limit is its own
    part, cut to it)."""
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for p in pieces:
        if buf and size + len(p) + 2 > limit:
            parts.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(p[:limit])
        size += len(buf[-1]) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts or [""]


_RECONCILE_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

RECONCILE_PROMPT = """Document: "{title}"
Section: {section}

This section was long, so it was read in consecutive parts. What each part says:
{parts}

Write the section's summary, 2-4 sentences, as one reading of the whole section: what it
actually says, specific -- methods, numbers, conclusions. Keep any exception, condition
or qualification a later part adds to an earlier one; do not let the opening stand for
the whole."""


async def reconcile_parts(c, model, title: str, section: str, outs: list[dict]) -> dict:
    """One section reading from its parts' readings: a reconciled summary, and the parts'
    claims, entities and categories pooled in order (duplicates dropped; categories by
    how many parts named them)."""

    def pooled(key: str) -> list[str]:
        seen: dict[str, None] = {}
        for o in outs:
            for x in o.get(key) or []:
                if isinstance(x, str) and x.strip() and x.strip().lower() not in {
                    k.lower() for k in seen
                }:
                    seen[x.strip()] = None
        return list(seen)

    votes: dict[str, int] = {}
    for o in outs:
        for x in o.get("categories") or []:
            if isinstance(x, str) and x.strip():
                votes[x.strip()] = votes.get(x.strip(), 0) + 1
    summaries = [str(o.get("summary") or "").strip() for o in outs]
    try:
        merged = await c.structured(
            model,
            RECONCILE_PROMPT.format(
                title=title,
                section=section,
                parts="\n".join(f"{k}. {x}" for k, x in enumerate(summaries, 1) if x),
            ),
            _RECONCILE_SCHEMA,
            system=prompts.SYSTEM_LIBRARIAN,
        )
        summary = str(merged.get("summary") or "").strip()
    except Exception:
        log.warning("reconciling %s failed; joining its parts", section, exc_info=True)
        summary = ""
    return {
        "summary": summary or " ".join(x for x in summaries if x),
        "entities": pooled("entities"),
        "claims": pooled("claims"),
        "categories": sorted(votes, key=lambda k: -votes[k])[:3],
        "parts": len(outs),
    }


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

        def outline(depth: int) -> str:
            return "\n".join(
                f"{'  ' * (s.level - 1)}- {s.title or '(untitled)'}"
                for s in sections
                if s.level <= depth and "(cont. " not in (s.title or "")
            )

        # a book's full outline overruns the card; its chapters alone do not
        toc = next((t for d in (9, 2, 1) if len(t := outline(d)) <= 4000), outline(1))

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
        pieces: dict[uuid.UUID, list[str]] = {}
        for s in sections:
            pieces[s.id] = list(
                (
                    await db.execute(
                        select(Chunk.text)
                        .where(Chunk.section_id == s.id, Chunk.kind == "text")
                        .order_by(Chunk.order_index)
                    )
                ).scalars()
            )
            bodies[s.id] = "\n\n".join(pieces[s.id])
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

                async def read(text_: str, where: str, _prev: str = previous) -> dict:
                    return await c.structured(
                        model,
                        prompts.SECTION_PROMPT.format(
                            title=doc.title,
                            orientation=one_liner,
                            previous=f"Previous section covered: {_prev}\n" if _prev else "",
                            section_path=where,
                            text=text_,
                            claims_guidance=genre_mod.CLAIMS_BY_GENRE.get(
                                genre, genre_mod.CLAIMS_BY_GENRE["other"]
                            ),
                            category_guidance=guidance,
                        ),
                        prompts.SECTION_SCHEMA,
                        system=prompts.SYSTEM_LIBRARIAN,
                    )

                where = s.path or s.title or "(untitled)"
                parts = section_parts(pieces.get(s.id) or [body], SECTION_CHARS)
                if len(parts) == 1:
                    out = await read(parts[0], where)
                else:
                    # A long section is read whole, in parts, and the parts reconciled --
                    # the exception or the decisive example is often at the end, where a
                    # first-6,000-characters read never looked.
                    outs = []
                    for k, part in enumerate(parts, start=1):
                        if gate:
                            await gate()
                        outs.append(await read(part, f"{where} (part {k} of {len(parts)})"))
                    out = await reconcile_parts(c, model, doc.title, where, outs)
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
        if len(joined) > SUMMARY_CHARS:
            # A book's section summaries overrun the prompt: digest them a stretch at a
            # time first, so the entry covers the last chapter as well as the first.
            joined = await _digest_stretches(c, model, doc.title, one_liner, summaries, gate)
        doc_out = await c.structured(
            model,
            prompts.DOCUMENT_SUMMARY_PROMPT.format(
                title=doc.title,
                orientation=one_liner,
                sections=joined[:SUMMARY_CHARS],
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
