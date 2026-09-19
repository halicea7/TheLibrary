"""Citation graph.

Structure for free: papers already tell you what they build on. Extracting references and
matching them against the corpus gives hub detection, claim lineage, and a Tier 2 promotion
signal without a single generation call.

Deliberately regex + fuzzy title matching rather than GROBID, which would add a Java
dependency for a first cut."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from itertools import pairwise

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.db.models import Chunk, Citation, Document

# "References", "Bibliography", "Works Cited" as a heading near the end of a document.
_REF_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|works cited)\s*$", re.IGNORECASE | re.MULTILINE
)
# Numbered entries: "[12] Author, Title..." or "12. Author, Title..."
_ENTRY = re.compile(r"(?:^|\n)\s*(?:\[(\d{1,3})\]|(\d{1,3})\.)\s+(?=[A-Z])")
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

# word_similarity(), not similarity(): a reference string is "Authors. Title. Venue Year",
# several times longer than the title, and plain trigram similarity penalises that length
# gap so heavily that nothing matches (measured: 0.51 for an exact title match, vs 1.00
# for word_similarity). word_similarity finds the best-matching extent instead.
MIN_TITLE_SIMILARITY = 0.72


@dataclass
class CitationResult:
    documents_scanned: int
    references_found: int
    matched: int


async def _document_text(db: AsyncSession, document_id: uuid.UUID) -> str:
    rows = (
        await db.execute(
            select(Chunk.text).where(Chunk.document_id == document_id).order_by(Chunk.order_index)
        )
    ).scalars()
    return "\n\n".join(rows)


def extract_references(body: str, *, max_refs: int = 300) -> list[str]:
    """Split the reference section into individual entries."""
    matches = list(_REF_HEADING.finditer(body))
    if not matches:
        return []
    # The reference list is the last such heading -- earlier ones are usually in the TOC.
    start = matches[-1].end()
    tail = body[start:]
    positions = [m.start() for m in _ENTRY.finditer(tail)]
    if len(positions) < 3:
        return []
    positions.append(len(tail))
    refs = []
    for a, b in pairwise(positions):
        entry = " ".join(tail[a:b].split())
        entry = re.sub(r"^\[?\d{1,3}\]?\.?\s*", "", entry)
        if 20 <= len(entry) <= 600:
            refs.append(entry)
    return refs[:max_refs]


async def match_references(
    db: AsyncSession, refs: list[str], corpus: dict[uuid.UUID, str], self_id: uuid.UUID
) -> list[tuple[str, uuid.UUID | None, float, str | None]]:
    """Match each reference against corpus titles using Postgres trigram similarity."""
    out: list[tuple[str, uuid.UUID | None, float, str | None]] = []
    for ref in refs:
        doi_match = _DOI.search(ref)
        doi = doi_match.group(0) if doi_match else None
        # Strip leading authors and trailing venue/year so the title dominates the match.
        probe = _YEAR.sub(" ", ref)
        row = (
            await db.execute(
                text("""
                select id, word_similarity(lower(title), lower(:probe)) as sim
                from document
                where id <> cast(:self as uuid)
                order by word_similarity(lower(title), lower(:probe)) desc
                limit 1
                """),
                {"probe": probe[:300], "self": str(self_id)},
            )
        ).first()
        if row and row.sim and float(row.sim) >= MIN_TITLE_SIMILARITY:
            out.append((ref, row.id, float(row.sim), doi))
        else:
            out.append((ref, None, float(row.sim) if row and row.sim else 0.0, doi))
    return out


async def build_citation_graph(db: AsyncSession, *, progress=None) -> CitationResult:
    docs = list((await db.execute(select(Document))).scalars())
    corpus = {d.id: d.title for d in docs}
    # A readings-only document (from a cartridge) has no text to mine references from;
    # the references it shipped with are all it will ever have. Keep those and re-match
    # them against whatever the corpus holds now.
    shipped: dict[uuid.UUID, list[str]] = {}
    for d in docs:
        if d.readings_only:
            shipped[d.id] = list(
                (
                    await db.execute(
                        select(Citation.raw_reference).where(Citation.citing_document_id == d.id)
                    )
                ).scalars()
            )
    await db.execute(delete(Citation))

    found = matched = 0
    for n, doc in enumerate(docs):
        if doc.readings_only:
            refs = shipped.get(doc.id, [])
        else:
            body = await _document_text(db, doc.id)
            refs = extract_references(body)
        found += len(refs)
        for ref, target, sim, doi in await match_references(db, refs, corpus, doc.id):
            db.add(
                Citation(
                    citing_document_id=doc.id,
                    matched_document_id=target,
                    raw_reference=ref[:1000],
                    doi=doi,
                    confidence=sim,
                )
            )
            if target:
                matched += 1
        if progress:
            await progress(n + 1, len(docs))
    await db.flush()
    return CitationResult(documents_scanned=len(docs), references_found=found, matched=matched)


async def hubs(db: AsyncSession, limit: int = 10) -> list[tuple[str, int]]:
    """Most-cited documents within the corpus -- a Tier 2 promotion signal."""
    rows = (
        await db.execute(
            text("""
            select d.title, count(*) as n
            from citation c join document d on d.id = c.matched_document_id
            where c.matched_document_id is not null
            group by d.title order by n desc limit :lim
            """),
            {"lim": limit},
        )
    ).all()
    return [(r.title, r.n) for r in rows]
