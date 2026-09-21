"""The shelf: a two-level taxonomy and one place per volume.

Tier 1 tags each document with up to four subjects, which is right for *finding* things
and wrong for *shelving* them: a security guide tagged Penetration Testing, Network
Security and Security Vulnerabilities sat on three shelves at once, and thirty flat
subjects with no order between them read as a mess.

So shelving is separate from tagging. There are top shelves (Cybersecurity) and
sub-shelves (Network Scanning), never a third level, and every volume sits on exactly one
sub-shelf (`document.shelf_id`). Tags stay many-to-many for filtering.

Two passes, both grammar-constrained:

* `build_taxonomy` -- given every subject in use, the model organises them into a small
  set of top shelves with sub-shelves under each, and says which existing subjects fold
  into which sub-shelf. Applied by setting `category.parent_id` and folding duplicates
  through the existing merge mechanism, so old names keep resolving.
* `place_document` -- given a volume's summary and the taxonomy, pick one sub-shelf. May
  add a sub-shelf under an existing top shelf when nothing fits; never a top shelf.

Tier 1 places each newly read volume; `reshelve` does the whole library."""

from __future__ import annotations

import logging
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.db.models import (
    Artifact,
    ArtifactKind,
    Category,
    Chunk,
    ChunkCategory,
    Document,
    DocumentCategory,
    TargetKind,
)
from library_agent.library import taxonomy
from library_agent.llm import providers
from library_agent.llm.client import LLM
from library_agent.llm.ollama import Ollama
from library_agent.reading.prompts import SYSTEM_LIBRARIAN

log = logging.getLogger(__name__)

MAX_TOP = 8
MAX_SUB = 7


def design_schema(max_top: int) -> dict[str, Any]:
    """Names only, every array capped. An open-ended array let grammar-constrained
    sampling run away (14k characters, never terminated); this grammar is finite."""
    return {
        "type": "object",
        "properties": {
            # Reasoning before the answer: without this the model names sub-shelves after
            # the tags it was handed and is done in two seconds.
            "notes": {"type": "string", "maxLength": 1500},
            "shelves": {
                "type": "array",
                "minItems": 2,
                "maxItems": max_top,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "maxLength": 40},
                        "sub_shelves": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": MAX_SUB,
                            "items": {"type": "string", "maxLength": 40},
                        },
                    },
                    "required": ["name", "sub_shelves"],
                },
            },
        },
        "required": ["notes", "shelves"],
    }


def assign_schema(subjects: list[str], sub_shelves: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "minItems": len(subjects),
                "maxItems": len(subjects),
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "enum": subjects},
                        "sub_shelf": {"type": "string", "enum": sub_shelves},
                    },
                    "required": ["subject", "sub_shelf"],
                },
            }
        },
        "required": ["assignments"],
    }


TAXONOMY_PROMPT = """You are organising a personal research library's shelf.

Below are the volumes, grouped by the subjects they were tagged with. Design TOP SHELVES
and SUB-SHELVES for them -- two levels, never three. A reader should find any volume by
walking Top shelf › Sub-shelf.

Rules:
- At most {max_top} top shelves, and fewer is better: broad, durable fields of
  knowledge, the way a bookshop labels its bays (for a different library that might be
  "Biology", "Economics", "Architecture"). Never two names for one field ("Security" and
  "Cybersecurity" are one shelf). Never an activity or an audience: no "Research",
  "Education", "Learning Resources", "Reference".
- Top shelves are DIFFERENT FIELDS, not slices of one field. If most of the library is
  one field, that is one top shelf with many sub-shelves -- never split it into "X
  Tools", "X Concepts", "X Methods" to fill the quota. The minority fields, however few
  volumes they have, get their own top shelf; do not lose them inside the majority.
- Each top shelf has 2 to {max_sub} sub-shelves: concrete, distinct from each other, and
  named for what the volumes actually cover (that other library might split Biology into
  "Genetics", "Ecology", "Neuroscience"). A sub-shelf must not restate its top shelf, and
  two sub-shelves must not be near-synonyms.
- Read the titles. Sub-shelves are the natural clusters you see in them. Each sub-shelf
  should hold a reasonable run of volumes -- roughly {per_sub} -- so a field with many
  volumes gets more sub-shelves and a field with a handful gets two.
- Two or three words each. Keep well-known acronyms as acronyms. Never name a sub-shelf
  after a single volume.
- The library holds several collections (listed below). A collection about a different
  field from the rest is its own field with its own top shelf, however the majority is
  shelved: an organisation's internal documentation does not go under the headings of a
  security handbook because the handbook is bigger.

First, in `notes`, walk through the titles and say what clusters you actually see. Then
name the shelves.

The collections:
{collections}

The volumes ({n} of them; a sample of the titles, drawn evenly from every collection):
{titles}

Tags previously applied to them, for reference only (inconsistent; do not copy them):
{subjects}
"""

ASSIGN_PROMPT = """Fold each old subject tag into the one sub-shelf it belongs to. Every tag
listed must appear exactly once.

Shelf (Top shelf › sub-shelves):
{shelf}

Tags to fold:
{tags}
"""

NEW = "(a new sub-shelf)"
# Tags folded per call: ~25 tokens a line keeps a batch well inside the output budget.
FOLD_BATCH = 120


def place_schema(tx: Taxonomy, *, allow_new: bool = True) -> dict[str, Any]:
    subs = sorted({s for subs in tx.tops.values() for s in subs})
    return {
        "type": "object",
        "properties": {
            "why": {"type": "string", "maxLength": 300},
            "sub_shelf": {"type": "string", "enum": [*subs, NEW] if allow_new else subs},
            "top_shelf": {"type": "string", "enum": sorted(tx.tops)},
            "new_sub_shelf_name": {"type": "string", "maxLength": 40},
        },
        # new_sub_shelf_name is legitimately empty most of the time; requiring it would
        # trip the placeholder guard.
        "required": ["why", "sub_shelf", "top_shelf"],
    }


PLACE_PROMPT = """Shelve one volume. Choose exactly one sub-shelf from the shelf below.

Shelf (Top shelf › sub-shelves):
{shelf}

Volume: {title}
Summary: {summary}
Subjects it was tagged with: {tags}
{collection}
In `why`, one sentence on what the volume is actually about and which sub-shelf that
points to. Then give the sub-shelf and its top shelf. Only if no sub-shelf reasonably fits,
choose "(a new sub-shelf)", name it in new_sub_shelf_name (concrete, two or three words,
in the style of its siblings) and say which top shelf it goes under; otherwise leave
new_sub_shelf_name empty. Never invent a top shelf.
"""


@dataclass
class Taxonomy:
    """Top shelf name -> sub-shelf names, plus ids."""

    tops: dict[str, list[str]] = field(default_factory=dict)
    ids: dict[str, uuid.UUID] = field(default_factory=dict)  # any name -> category id

    def render(self) -> str:
        return "\n".join(f"- {t} › " + " · ".join(subs) for t, subs in self.tops.items())

    def empty(self) -> bool:
        return not self.tops


async def load_taxonomy(db: AsyncSession) -> Taxonomy:
    # Structure is parent_id, not the canonical flag: a shelf that holds volumes is a
    # shelf whatever its tagging history.
    rows = list((await db.execute(select(Category))).scalars())
    by_id = {c.id: c for c in rows}
    tx = Taxonomy()
    for c in rows:
        if c.parent_id is None and any(x.parent_id == c.id for x in rows):
            tx.tops[c.name] = []
            tx.ids[c.name] = c.id
    for c in rows:
        if c.parent_id and c.parent_id in by_id and by_id[c.parent_id].name in tx.tops:
            c.canonical = True  # it is on the shelf; it must resolve by name again
            tx.tops[by_id[c.parent_id].name].append(c.name)
            tx.ids[c.name] = c.id
    for subs in tx.tops.values():
        subs.sort()
    return tx


async def _subjects_in_use(db: AsyncSession) -> list[tuple[str, int, list[str]]]:
    rows = (
        await db.execute(
            select(Category.name, func.count(DocumentCategory.document_id))
            .join(DocumentCategory, DocumentCategory.category_id == Category.id)
            .where(Category.canonical.is_(True))
            .group_by(Category.id)
            .order_by(func.count(DocumentCategory.document_id).desc())
        )
    ).all()
    out = []
    for name, n in rows:
        titles = list(
            (
                await db.execute(
                    select(Document.title)
                    .join(DocumentCategory, DocumentCategory.document_id == Document.id)
                    .join(Category, Category.id == DocumentCategory.category_id)
                    .where(Category.name == name)
                    .order_by(Document.title)
                    .limit(6)
                )
            ).scalars()
        )
        out.append((name, n, titles))
    return out


async def _fold(db: AsyncSession, loser: Category, winner: Category) -> None:
    """Merge one category into another: move memberships, keep the old name resolving."""
    if loser.id == winner.id:
        return
    for model, key in (
        (DocumentCategory, DocumentCategory.document_id),
        (ChunkCategory, ChunkCategory.chunk_id),
    ):
        existing = set(
            (await db.execute(select(key).where(model.category_id == winner.id))).scalars()
        )
        movers = list(
            (await db.execute(select(key).where(model.category_id == loser.id))).scalars()
        )
        for owner in movers:
            if owner in existing:
                continue
            db.add(model(**{key.key: owner, "category_id": winner.id}))
            existing.add(owner)
        await db.execute(
            model.__table__.delete().where(model.category_id == loser.id)  # type: ignore[attr-defined]
        )
    await db.execute(
        update(Document).where(Document.shelf_id == loser.id).values(shelf_id=winner.id)
    )
    loser.canonical = False
    loser.parent_id = None
    winner.merged_from = sorted(
        set(winner.merged_from or []) | {loser.name} | set(loser.merged_from or [])
    )
    await db.flush()


_BANNED_TOPS = {
    "research",
    "education",
    "reference",
    "references",
    "resources",
    "general",
    "other",
    "misc",
    "tutorials",
    "preparation",
    "guides",
    "cheat",
    "sheets",
    "commands",
    "concepts",
    "methodologies",
    "methods",
    "techniques",
    "tools",
    "frameworks",
}


def _top_names(out: dict[str, Any]) -> list[str]:
    return [taxonomy.normalize_name(str(s.get("name") or "")) for s in out.get("shelves") or []]


def _taxonomy_is_sane(out: dict[str, Any], max_top: int) -> bool:
    """Reject the shapes the model reaches for when it is not thinking: two names for one
    field, activity shelves, or a bay with a single sub-shelf."""
    tops = _top_names(out)
    if not tops or len(tops) > max_top + 1:
        return False
    # Schema-shaped filler: "Top Shelf 1 › Sub-Shelf 1 · Sub-Shelf 2".
    every = tops + [str(k) for s in out.get("shelves") or [] for k in s.get("sub_shelves") or []]
    if any(re.search(r"\b(shelf|category|topic)\s*\d*$", n.strip(), re.IGNORECASE) for n in every):
        return False
    low = [t.lower() for t in tops]
    for i, a in enumerate(low):
        if any(w in _BANNED_TOPS for w in re.split(r"[\s&/-]+", a)):
            return False
        for j, b in enumerate(low):
            if i != j and (a in b or b in a):
                return False
            # "Security Vulnerabilities" and "Security Tools" are one field sliced in two.
            shared = {w for w in a.split() if len(w) > 4} & {w for w in b.split() if len(w) > 4}
            if i != j and shared:
                return False
    seen: set[str] = set()
    for s in out.get("shelves") or []:
        if len(s.get("sub_shelves") or []) < 2:
            return False
        for k in s.get("sub_shelves") or []:
            name = taxonomy.normalize_name(str(k)).lower()
            # A sub-shelf named like a top shelf, or like a sub-shelf on another top,
            # is a name the placer can point at and the shelf cannot resolve: half of
            # one run's placements named "Network Services", which was both.
            if name in low or name in seen:
                return False
            seen.add(name)
    return True


DESIGN_KEY = "taxonomy_design"
# A reshelve is due when the read collection has grown by this much since the top
# shelves were designed: a quarter, and at least twenty volumes.
REDESIGN_FRACTION = 0.25
REDESIGN_MIN = 20


async def record_design(db: AsyncSession) -> None:
    """Remember when the top shelves were designed and on how many read volumes."""
    from datetime import UTC, datetime

    from library_agent.db.models import LibraryMeta

    n = (
        await db.execute(select(func.count()).select_from(Document).where(Document.tier >= 1))
    ).scalar() or 0
    row = await db.get(LibraryMeta, DESIGN_KEY)
    value = {"at": datetime.now(UTC).isoformat(), "read_volumes": n}
    if row:
        row.value = value
    else:
        db.add(LibraryMeta(key=DESIGN_KEY, value=value))
    await db.flush()


async def shelf_health(db: AsyncSession) -> dict[str, Any]:
    """Is a reshelve due? Three signals: read volumes with no place; no top shelves at
    all while there is something to shelve; and the collection having grown past the
    shelves it was designed for. The reason is a sentence for the button's tooltip."""
    from library_agent.db.models import Job, JobState, LibraryMeta

    read = (
        await db.execute(select(func.count()).select_from(Document).where(Document.tier >= 1))
    ).scalar() or 0
    unshelved = (
        await db.execute(
            select(func.count())
            .select_from(Document)
            .where(Document.tier >= 1, Document.shelf_id.is_(None))
        )
    ).scalar() or 0
    tops = (
        await db.execute(
            select(func.count())
            .select_from(Category)
            .where(Category.parent_id.is_(None), Category.id.in_(select(Category.parent_id)))
        )
    ).scalar() or 0
    in_hand = (
        await db.execute(
            select(func.count())
            .select_from(Job)
            .where(Job.kind == "reshelve", Job.state.in_([JobState.QUEUED, JobState.RUNNING]))
        )
    ).scalar() or 0
    meta = await db.get(LibraryMeta, DESIGN_KEY)
    designed_on = (meta.value or {}).get("read_volumes") if meta else None
    grown = read - designed_on if designed_on is not None else None
    out: dict[str, Any] = {
        "read": read,
        "unshelved": unshelved,
        "top_shelves": tops,
        "designed_on": designed_on,
        "grown": grown,
        "in_hand": bool(in_hand),
        "needed": False,
        "reason": "",
    }
    if in_hand or read == 0:
        return out
    if tops == 0:
        out.update(needed=True, reason=f"{read} read volumes and no shelves yet")
    elif unshelved >= max(3, read // 20):
        out.update(needed=True, reason=f"{unshelved} read volumes have no place on the shelf")
    elif grown is not None and grown >= max(REDESIGN_MIN, int(designed_on * REDESIGN_FRACTION)):
        out.update(
            needed=True,
            reason=f"the shelves were designed for {designed_on} volumes; there are {read} now",
        )
    elif designed_on is None and read >= REDESIGN_MIN:
        out.update(needed=True, reason="the shelves have not been designed on this collection")
    return out


DESIGN_TITLES = 300


async def _design_sample(db: AsyncSession) -> tuple[list[str], str, str]:
    """The titles the designer reads, and a line on each collection.

    The first three hundred titles in alphabetical order were all one cartridge's when
    that cartridge was a thousand volumes, and the design never saw the other three
    hundred and ninety: an internal wiki went under "System Exploitation". Every
    collection -- each cartridge, and the volumes shelved directly -- gets a share of the
    sample with a floor, taken evenly through its titles, and the prompt is told what the
    collections are."""
    rows = (
        await db.execute(
            text("""
            select d.title, c.name as coll, c.genre as cgenre, c.description as about, d.genre
            from document d
            left join cartridge_document cd on cd.document_id = d.id
            left join cartridge c on c.id = cd.cartridge_id
            where d.tier >= 1
            order by d.title
            """)
        )
    ).all()
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r.coll or "", []).append(r)
    all_titles = [r.title for r in rows]
    total = len(rows)
    floor = min(40, DESIGN_TITLES // max(1, len(groups)))
    picked: list[str] = []
    lines: list[str] = []
    for coll, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        share = max(floor, round(DESIGN_TITLES * len(members) / max(1, total)))
        share = min(share, len(members))
        step = max(1, len(members) // share)
        picked.extend(m.title for m in members[::step][:share])
        # The cartridge's genre if it has one, else the commonest among its volumes.
        cg = next((m.cgenre for m in members if m.cgenre), None)
        counts = Counter(m.genre for m in members if m.genre)
        genre = cg or (counts.most_common(1)[0][0] if counts else None)
        what = f"{len(members)} volumes" + (
            f", mostly {genre}" if genre and not cg else f", {genre}" if genre else ""
        )
        about = next((m.about for m in members if m.about), None)
        line = f"- {coll} (a cartridge; {what})" if coll else f"- shelved directly ({what})"
        if about:
            # The maker's own account of the collection outranks any guess from titles.
            line += f": {' '.join(about.split())[:600]}"
        lines.append(line)
    titles = "\n".join(f"- {x[:70]}" for x in picked)
    return all_titles, titles, "\n".join(lines)


async def build_taxonomy(
    db: AsyncSession, *, client: Ollama | None = None, progress=None
) -> Taxonomy:
    """One model call organises every subject in use into two levels, then it is applied."""
    subjects = await _subjects_in_use(db)
    if not subjects:
        return Taxonomy()
    listing = ", ".join(f"{name} ({n})" for name, n, _ in subjects)
    all_titles, titles, collections = await _design_sample(db)
    # A small library gets a small shelf: 170 volumes want four bays, not eight.
    max_top = max(2, min(MAX_TOP, len(all_titles) // 40))
    per_sub = f"{max(3, len(all_titles) // 12)} to {max(8, len(all_titles) // 5)}"
    names = [name for name, _, _ in subjects]
    own = client is None
    c = client or LLM()
    try:
        design: dict[str, Any] = {}
        for attempt in range(3):
            if progress:
                await progress(0, 1, "designing the shelves")
            design = await c.structured(
                providers.model_for("threads"),
                TAXONOMY_PROMPT.format(
                    max_top=max_top,
                    max_sub=MAX_SUB,
                    per_sub=per_sub,
                    n=len(all_titles),
                    titles=titles,
                    subjects=listing,
                    collections=collections,
                ),
                design_schema(max_top),
                system=SYSTEM_LIBRARIAN,
                instructions=TAXONOMY_PROMPT,
                temperature=0.1 + 0.2 * attempt,
                think=True,
            )
            if _taxonomy_is_sane(design, max_top):
                log.info("taxonomy attempt %d accepted: %s", attempt, _top_names(design))
                break
            log.info("taxonomy attempt %d rejected: %s", attempt, _top_names(design))
        design["shelves"] = (design.get("shelves") or [])[:max_top]
        shelf_lines = []
        subs: list[str] = []
        for s in design.get("shelves") or []:
            kids = [taxonomy.normalize_name(str(x)) for x in s.get("sub_shelves") or []]
            kids = [k for k in kids if k]
            subs.extend(kids)
            shelf_lines.append(
                f"- {taxonomy.normalize_name(str(s.get('name') or ''))} › " + " · ".join(kids)
            )
        subs = list(dict.fromkeys(subs))
        folded: dict[str, str] = {}
        if subs:
            # One line of output per tag, so a big shelf is folded in batches: a
            # thousand tags in one call ran past the output budget and came back as
            # half a JSON document.
            for i in range(0, len(names), FOLD_BATCH):
                batch = names[i : i + FOLD_BATCH]
                if progress:
                    await progress(i, len(names), "folding the old subjects in")
                assign = await c.structured(
                    providers.model_for("threads"),
                    ASSIGN_PROMPT.format(
                        shelf="\n".join(shelf_lines), tags="\n".join(f"- {n}" for n in batch)
                    ),
                    assign_schema(batch, subs),
                    system=SYSTEM_LIBRARIAN,
                    temperature=0.1,
                    num_predict=4000,  # one line per tag; the grammar is finite
                )
                got = assign.get("assignments") or []
                log.info("fold: %d assignments for %d tags", len(got), len(batch))
                for a in got:
                    folded.setdefault(str(a.get("subject")), str(a.get("sub_shelf")))
        out = {
            "shelves": [
                {
                    "name": s.get("name"),
                    "sub_shelves": [
                        {
                            "name": k,
                            "absorbs": [
                                n
                                for n, sub in folded.items()
                                if sub == taxonomy.normalize_name(str(k))
                            ],
                        }
                        for k in s.get("sub_shelves") or []
                    ],
                }
                for s in design.get("shelves") or []
            ]
        }
        log.info(
            "fold: %d tags absorbed; unmatched sub-shelf names: %s",
            sum(len(k["absorbs"]) for s in out["shelves"] for k in s["sub_shelves"]),
            sorted(
                set(folded.values())
                - {
                    taxonomy.normalize_name(str(k))
                    for s in design.get("shelves") or []
                    for k in s.get("sub_shelves") or []
                }
            ),
        )
    finally:
        if own:
            await c.aclose()

    # Clear the old structure; every canonical category is re-homed below.
    await db.execute(update(Category).values(parent_id=None))
    await record_design(db)
    tx = Taxonomy()
    absorbed: set[uuid.UUID] = set()
    top_names = {
        taxonomy.normalize_name(str(s.get("name") or ""))
        for s in (out.get("shelves") or [])[:MAX_TOP]
    }
    for shelf in (out.get("shelves") or [])[:MAX_TOP]:
        top = await taxonomy.get_or_create(db, str(shelf.get("name") or ""))
        if not top:
            continue
        top.parent_id = None
        tx.tops[top.name] = []
        tx.ids[top.name] = top.id
        for sub in (shelf.get("sub_shelves") or [])[:MAX_SUB]:
            name = taxonomy.normalize_name(str(sub.get("name") or ""))
            if name in top_names or name in tx.ids:
                # Belt and braces after the sanity check: a name that is also a top
                # shelf, or already a sub-shelf elsewhere, is not a place.
                log.info("sub-shelf %r dropped: it names a top shelf or another sub-shelf", name)
                continue
            child = await taxonomy.get_or_create(db, name)
            if not child or child.id == top.id:
                continue
            child.parent_id = top.id
            child.canonical = True
            for old_name in sub.get("absorbs") or []:
                old = await taxonomy.get_or_create(db, str(old_name))
                if old and old.id not in (child.id, top.id) and old.name not in tx.tops:
                    await _fold(db, old, child)
                    absorbed.add(old.id)
            if child.name not in tx.tops[top.name]:
                tx.tops[top.name].append(child.name)
                tx.ids[child.name] = child.id
    # A top shelf is a shelf, not a tag: nothing should sit on it directly.
    await db.execute(
        update(Document)
        .where(Document.shelf_id.in_([tx.ids[t] for t in tx.tops] or [uuid.uuid4()]))
        .values(shelf_id=None)
    )
    await db.flush()
    return tx


async def _collection_note(db: AsyncSession, doc: Document) -> str:
    """The cartridge a volume belongs to, with its maker's description when there is
    one: a page of an internal wiki reads like anything else until you know that."""
    rows = (
        await db.execute(
            text("""
            select c.name, c.description from cartridge c
            join cartridge_document cd on cd.cartridge_id = c.id
            where cd.document_id = :d order by c.name
            """),
            {"d": doc.id},
        )
    ).all()
    if not rows:
        return ""
    parts = [
        f"{name}" + (f" — {' '.join(about.split())[:400]}" if about else "") for name, about in rows
    ]
    return "From the collection: " + "; ".join(parts) + "\n"


async def _summary_and_tags(db: AsyncSession, doc: Document) -> tuple[str, list[str]]:
    summary = (
        await db.execute(
            select(Artifact.text).where(
                Artifact.kind == ArtifactKind.DOCUMENT_SUMMARY,
                Artifact.target_kind == TargetKind.DOCUMENT,
                Artifact.target_id == doc.id,
            )
        )
    ).scalars().first() or ""
    if not summary:
        # A one-page stub may never have earned a summary; its opening is better than a
        # bare title, which had "Git" shelved under Information Retrieval.
        summary = (
            await db.execute(
                select(Chunk.text)
                .where(Chunk.document_id == doc.id)
                .order_by(Chunk.order_index)
                .limit(1)
            )
        ).scalar() or ""
        summary = " ".join(summary.split())[:1200]
    tags = list(
        (
            await db.execute(
                select(Category.name)
                .join(DocumentCategory, DocumentCategory.category_id == Category.id)
                .where(DocumentCategory.document_id == doc.id)
                .where(Category.id != doc.shelf_id)  # its current shelf is not evidence
            )
        ).scalars()
    )
    return summary, tags


async def place_document(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    client: Ollama | None = None,
    tx: Taxonomy | None = None,
    allow_new: bool = True,
) -> Category | None:
    """Put one volume on one sub-shelf. Returns it, or None when there is no shelf yet.

    The model's answer is checked against the shelf, and an answer that names nothing on
    it (a top shelf given as the sub-shelf, a new-shelf name that is not a name) is asked
    again once, warmer. On a big shelf about one answer in twelve needed that; without
    the second ask those volumes were left with no place and the tracer had to say so."""
    tx = tx or await load_taxonomy(db)
    if tx.empty():
        return None
    doc = (await db.execute(select(Document).where(Document.id == document_id))).scalar_one()
    summary, tags = await _summary_and_tags(db, doc)
    collection = await _collection_note(db, doc)
    own = client is None
    c = client or LLM()
    try:
        out: dict[str, Any] = {}
        sub: Category | None = None
        for temperature in (0.2, 0.5):
            out = await c.structured(
                providers.model_for("threads"),
                PLACE_PROMPT.format(
                    shelf=tx.render(),
                    title=doc.title,
                    summary=summary[:1500] or "(not yet read)",
                    tags=", ".join(tags) or "none",
                    collection=collection,
                ),
                place_schema(tx, allow_new=allow_new),
                system=SYSTEM_LIBRARIAN,
                instructions=PLACE_PROMPT,
                temperature=temperature,
            )
            sub = await _resolve_place(db, tx, out)
            if sub:
                break
            log.info("no shelf for %s at %.1f: %r", doc.title[:40], temperature, out)
    finally:
        if own:
            await c.aclose()
    if not sub:
        log.warning("no shelf for %s after two asks: %r", doc.title[:40], out)
        return None
    doc.shelf_id = sub.id
    return await _finish_place(db, doc, sub)


async def _resolve_place(db: AsyncSession, tx: Taxonomy, out: dict[str, Any]) -> Category | None:
    """The sub-shelf an answer names, or None when it names nothing on the shelf."""
    top_name = str(out.get("top_shelf") or "")
    choice = str(out.get("sub_shelf") or "")
    top = top_name if top_name in tx.tops else None
    sub: Category | None = None
    if choice != NEW and choice in tx.ids and choice not in tx.tops:
        sub = await db.get(Category, tx.ids[choice])
    elif choice == NEW and top:
        new_name = taxonomy.normalize_name(str(out.get("new_sub_shelf_name") or ""))
        if 3 <= len(new_name) <= 40 and new_name not in tx.tops:
            sub = await taxonomy.get_or_create(db, new_name)
            if sub and sub.parent_id is None:
                sub.parent_id = tx.ids[top]
                tx.tops[top].append(sub.name)
                tx.ids[sub.name] = sub.id
    return sub


async def _finish_place(db: AsyncSession, doc: Document, sub: Category) -> Category:
    # The shelf is also a tag, so filtering by it finds the volume.
    have = set(
        (
            await db.execute(
                select(DocumentCategory.category_id).where(DocumentCategory.document_id == doc.id)
            )
        ).scalars()
    )
    if sub.id not in have:
        db.add(DocumentCategory(document_id=doc.id, category_id=sub.id))
    await db.flush()
    return sub


SPLIT_PROMPT = """One sub-shelf of a library has grown too crowded to browse. Split it.

Top shelf: {top}
Crowded sub-shelf: {sub} ({n} volumes)
Its neighbours on the same top shelf (do not duplicate these): {siblings}

The volumes on it:
{titles}

Design {lo} to {hi} new sub-shelves that these volumes divide into naturally, by what the
titles cover. Concrete, two or three words each, distinct from each other and from the
neighbours. A volume that fits nowhere else may need a broad one, but do not name a
sub-shelf after the crowded one you are replacing, and no catch-all ("Other …", "… Types",
"General …"): every volume must have a specific home. First, in `notes`, say what clusters
you see; then name them.
"""


def split_schema(lo: int, hi: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "notes": {"type": "string", "maxLength": 1200},
            "sub_shelves": {
                "type": "array",
                "minItems": lo,
                "maxItems": hi,
                "items": {"type": "string", "maxLength": 40},
            },
        },
        "required": ["notes", "sub_shelves"],
    }


CROWDED = 30


async def split_crowded(
    db: AsyncSession, tx: Taxonomy, *, client: Ollama, progress=None, gate=None
) -> int:
    """Re-design any sub-shelf holding more than CROWDED volumes from its own titles, then
    re-place those volumes among the new sub-shelves. Returns how many were re-placed."""
    moved = 0
    for top, subs in list(tx.tops.items()):
        for sub in list(subs):
            sid = tx.ids[sub]
            docs = list(
                (
                    await db.execute(
                        select(Document).where(Document.shelf_id == sid).order_by(Document.title)
                    )
                ).scalars()
            )
            if len(docs) <= CROWDED:
                continue
            n = len(docs)
            lo, hi = 3, max(4, min(8, -(-n // 20)))
            out = await client.structured(
                providers.model_for("threads"),
                SPLIT_PROMPT.format(
                    top=top,
                    sub=sub,
                    n=n,
                    siblings=", ".join(s for s in subs if s != sub) or "none",
                    titles="\n".join(f"- {d.title[:70]}" for d in docs[:120]),
                    lo=lo,
                    hi=hi,
                ),
                split_schema(lo, hi),
                system=SYSTEM_LIBRARIAN,
                instructions=SPLIT_PROMPT,
                num_predict=2000,
                think=True,
            )
            names = [taxonomy.normalize_name(str(x)) for x in out.get("sub_shelves") or []]
            # A catch-all ("Web Attack Types", "Other Exploits") just recreates the lump.
            catchall = {"types", "other", "others", "general", "misc", "various", "miscellaneous"}
            names = [
                x
                for x in dict.fromkeys(names)
                if x and x not in tx.tops and x != sub and not (set(x.lower().split()) & catchall)
            ]
            if len(names) < 2:
                log.info("split of %s produced nothing usable", sub)
                continue
            for name in names:
                cat = await taxonomy.get_or_create(db, name)
                # A proposed name can resolve, through an old fold, to the very shelf
                # being split; that would put the volumes straight back on it.
                if not cat or cat.id == sid or cat.name in tx.tops:
                    continue
                if cat.parent_id is None:
                    cat.parent_id = tx.ids[top]
                if cat.name not in tx.tops[top]:
                    tx.tops[top].append(cat.name)
                    tx.ids[cat.name] = cat.id
            names = [n for n in names if n in tx.tops[top]]
            if len(names) < 2:
                continue
            # The crowded shelf comes off the rack; its volumes get re-placed below.
            tx.tops[top].remove(sub)
            crowded = await db.get(Category, sid)
            if crowded:
                crowded.parent_id = None
            await db.flush()
            # Re-place within this top shelf only.
            local = Taxonomy(tops={top: tx.tops[top]}, ids=dict(tx.ids))
            for i, d in enumerate(docs):
                if gate:
                    await gate()
                d.shelf_id = None
                try:
                    if await place_document(db, d.id, client=client, tx=local, allow_new=False):
                        moved += 1
                except Exception:
                    log.warning("re-placing %s failed", d.id, exc_info=True)
                if progress:
                    await progress(i + 1, n, "splitting crowded shelves")
            # Anything still unplaced keeps a home: the first new sub-shelf.
            fallback = tx.ids[names[0]]
            await db.execute(
                update(Document)
                .where(Document.id.in_([d.id for d in docs]), Document.shelf_id.is_(None))
                .values(shelf_id=fallback)
            )
            tx.tops[top] = local.tops[top]
            tx.ids.update(local.ids)
            await db.flush()
    return moved


SPARSE = 3


async def merge_sparse(db: AsyncSession, tx: Taxonomy, *, client: Ollama, gate=None) -> int:
    """A sub-shelf with one or two volumes is a label, not a shelf. Dissolve it and
    re-place its volumes among the siblings (no new sub-shelves allowed this time)."""
    moved = 0
    for top, subs in list(tx.tops.items()):
        counts = {}
        for sub in subs:
            counts[sub] = (
                await db.execute(select(func.count()).where(Document.shelf_id == tx.ids[sub]))
            ).scalar_one()
        for sub in sorted(subs, key=lambda s: counts[s]):
            if counts[sub] >= SPARSE or len(tx.tops[top]) <= 2:
                continue
            docs = list(
                (
                    await db.execute(select(Document).where(Document.shelf_id == tx.ids[sub]))
                ).scalars()
            )
            tx.tops[top].remove(sub)
            cat = await db.get(Category, tx.ids[sub])
            if cat:
                cat.parent_id = None
            local = Taxonomy(tops={top: tx.tops[top]}, ids=dict(tx.ids))
            for d in docs:
                if gate:
                    await gate()
                d.shelf_id = None
                try:
                    placed = await place_document(
                        db, d.id, client=client, tx=local, allow_new=False
                    )
                except Exception:  # noqa: BLE001
                    placed = None
                if not placed:
                    d.shelf_id = tx.ids[tx.tops[top][0]]
                moved += 1
            await db.flush()
    return moved


@dataclass
class ReshelveResult:
    top_shelves: int = 0
    sub_shelves: int = 0
    placed: int = 0
    unplaced: int = 0
    split_moved: int = 0
    merged_moved: int = 0


async def reshelve(
    db: AsyncSession,
    *,
    client: Ollama | None = None,
    rebuild: bool = True,
    progress=None,
    gate=None,
    category_id: uuid.UUID | None = None,
) -> ReshelveResult:
    """Build (or keep) the taxonomy, then place every read volume that needs a place.

    With `category_id`, keep the taxonomy and re-place just that shelf's volumes -- the
    remedy when one sub-shelf has collected things that do not belong on it."""
    if category_id:
        rebuild = False
    own = client is None
    c = client or LLM()
    res = ReshelveResult()
    try:
        tx = await load_taxonomy(db)
        if rebuild or tx.empty():
            tx = await build_taxonomy(db, client=c, progress=progress)
        res.top_shelves = len(tx.tops)
        res.sub_shelves = sum(len(s) for s in tx.tops.values())
        if tx.empty():
            return res
        q = select(Document.id).where(Document.tier >= 1)
        if category_id:
            q = q.where(Document.shelf_id.in_(await expand_category_ids(db, [category_id])))
        elif not rebuild:
            # Unshelved, or on a shelf that is no longer under a top shelf.
            homed = select(Category.id).where(Category.parent_id.is_not(None))
            q = q.where(Document.shelf_id.is_(None) | Document.shelf_id.not_in(homed))
        ids = list((await db.execute(q.order_by(Document.added_at))).scalars())
        left: list[uuid.UUID] = []
        for i, did in enumerate(ids):
            if gate:
                await gate()
            if progress and i == 0:
                await progress(0, len(ids), "placing volumes")
            try:
                placed = await place_document(db, did, client=c, tx=tx)
            except Exception:
                log.warning("placing %s failed", did, exc_info=True)
                placed = None
            if placed:
                res.placed += 1
            else:
                left.append(did)
            if progress:
                await progress(i + 1, len(ids), "placing volumes")
        # Then even the shelf out: split the crowded (a split can leave one lump behind,
        # so twice), then dissolve the sparse.
        for _ in range(2):
            moved = await split_crowded(db, tx, client=c, progress=progress, gate=gate)
            res.split_moved += moved
            if not moved:
                break
        res.merged_moved = await merge_sparse(db, tx, client=c, gate=gate)
        # The shelf is different now; whatever found no place gets one more look at it.
        # What is still homeless after that is the tracer's to report.
        for i, did in enumerate(left):
            if gate:
                await gate()
            if progress:
                await progress(i, len(left), "placing the rest")
            try:
                placed = await place_document(db, did, client=c, tx=tx)
            except Exception:
                log.warning("placing %s failed", did, exc_info=True)
                placed = None
            res.placed += 1 if placed else 0
            res.unplaced += 0 if placed else 1
        res.sub_shelves = sum(len(s) for s in tx.tops.values())
        return res
    finally:
        if own:
            await c.aclose()


async def expand_category_ids(db: AsyncSession, ids: list[uuid.UUID]) -> list[uuid.UUID]:
    """A top shelf stands for everything under it: scoping to one scopes to its children."""
    if not ids:
        return ids
    kids = list(
        (await db.execute(select(Category.id).where(Category.parent_id.in_(ids)))).scalars()
    )
    return list({*ids, *kids})
