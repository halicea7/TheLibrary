"""Classification: a scale of levels, markings found in the text, and what they govern.

The scale is yours: a US-government preset (U / CUI / C / S / TS, with banner lines such as
SECRET//NOFORN and portion marks such as "(S)" at the start of a paragraph), a company
preset (Public / Internal / Confidential / Restricted), or your own levels. It lives in
~/.library-agent/classification.json.

A volume's level comes, strongest first, from a level set by hand, from markings in its
text, from the cartridge it arrived in, or from the scale's default. A passage carries its
own portion mark where the document has them, else its volume's level.

What the levels govern: output is marked (each paragraph takes the highest level of what
it draws on -- and, in strict mode, uncited synthesis takes the highest level the model
saw); a conversation can set a ceiling, above which nothing is retrieved; nothing above
the remote ceiling is ever sent to a remote model; a cartridge export leaves out volumes
above the export ceiling.

This labels and enforces inside the library. It is not an accredited system for handling
classified national-security information."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from library_agent.config import settings


@dataclass
class Level:
    id: str  # stable key stored on volumes and passages, e.g. "S"
    label: str  # "Secret"
    short: str  # the portion mark: "S" -> "(S)"
    colour: str  # the banner colour
    # Banner lines and portion marks that mean this level (regexes, case-sensitive).
    banners: list[str] = field(default_factory=list)
    portions: list[str] = field(default_factory=list)


US = [
    Level("U", "Unclassified", "U", "#2e7d32", [r"^UNCLASSIFIED(//[A-Z /]+)?$"], [r"U"]),
    Level(
        "CUI",
        "Controlled Unclassified",
        "CUI",
        "#6a3fa0",
        [r"^CUI(//[A-Z /-]+)?$", r"^CONTROLLED UNCLASSIFIED INFORMATION$"],
        [r"CUI(//[A-Z-]+)?"],
    ),
    Level("C", "Confidential", "C", "#1565c0", [r"^CONFIDENTIAL(//[A-Z /]+)?$"], [r"C(//[A-Z]+)?"]),
    Level("S", "Secret", "S", "#c62828", [r"^SECRET(//[A-Z /]+)?$"], [r"S(//[A-Z]+)?"]),
    Level("TS", "Top Secret", "TS", "#ef6c00", [r"^TOP SECRET(//[A-Z /]+)?$"], [r"TS(//[A-Z]+)?"]),
]
COMPANY = [
    Level("public", "Public", "P", "#2e7d32", [r"^PUBLIC$"], []),
    Level("internal", "Internal", "I", "#3d7a69", [r"^INTERNAL( USE ONLY)?$"], []),
    Level("confidential", "Confidential", "CONF", "#9a6b1f", [r"^(COMPANY )?CONFIDENTIAL$"], []),
    Level("restricted", "Restricted", "R", "#a3432a", [r"^(STRICTLY )?RESTRICTED$"], []),
]
PRESETS = {"us": US, "company": COMPANY}


@dataclass
class Scale:
    scheme: str = "company"  # "us", "company", or "custom"
    levels: list[Level] = field(default_factory=lambda: list(COMPANY))
    default: str = "internal"  # a volume with no marking and nothing set
    remote_ceiling: str = "internal"  # the highest level a remote model may ever see
    export_ceiling: str = "internal"  # the highest level a cartridge may carry out
    mode: str = "strict"  # strict: uncited synthesis takes the highest level in context
    marking: bool = True  # portion marks and banners on answers and compositions

    def rank(self, level_id: str | None) -> int:
        ids = [lv.id for lv in self.levels]
        return (
            ids.index(level_id)
            if level_id in ids
            else ids.index(self.default)
            if self.default in ids
            else 0
        )

    def at_or_below(self, level_id: str) -> list[str]:
        r = self.rank(level_id)
        return [lv.id for lv in self.levels if self.rank(lv.id) <= r]

    def highest(self, ids) -> str:
        ids = [i for i in ids if i]
        return max(ids, key=self.rank) if ids else self.default

    def level(self, level_id: str | None) -> Level:
        return self.levels[self.rank(level_id)]

    def public(self) -> dict:
        return asdict(self)


def _path() -> Path:
    override = os.environ.get("LIBRARY_CLASSIFICATION_FILE")
    return Path(override) if override else settings().storage_dir.parent / "classification.json"


def preset(scheme: str) -> Scale:
    if scheme == "us":
        return Scale(
            scheme="us", levels=list(US), default="U", remote_ceiling="U", export_ceiling="U"
        )
    return Scale()


def load() -> Scale:
    p = _path()
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return Scale()
    levels = [Level(**lv) for lv in raw.get("levels") or []] or list(
        PRESETS.get(raw.get("scheme"), COMPANY)
    )
    s = Scale(
        scheme=raw.get("scheme", "company"),
        levels=levels,
        default=raw.get("default") or levels[0].id,
        remote_ceiling=raw.get("remote_ceiling") or levels[0].id,
        export_ceiling=raw.get("export_ceiling") or levels[0].id,
        mode=raw.get("mode", "strict"),
        marking=raw.get("marking", True),
    )
    return s


def save(s: Scale) -> None:
    ids = [lv.id for lv in s.levels]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("levels need distinct ids")
    for k in ("default", "remote_ceiling", "export_ceiling"):
        if getattr(s, k) not in ids:
            raise ValueError(f"{k} must be one of the levels")
    for lv in s.levels:
        for pat in (*lv.banners, *lv.portions):
            re.compile(pat)
    p = _path()
    p.write_text(json.dumps(s.public(), indent=2))


# ----------------------------------------------------------------- markings in the text


def _banner_line(line: str) -> str:
    return " ".join(line.strip().strip("*#_-= ").split())


def detect_document(s: Scale, text: str) -> str | None:
    """The highest level any banner line in the text marks -- a line that is nothing but
    a marking ("SECRET//NOFORN", "COMPANY CONFIDENTIAL"), never a word mid-sentence."""
    found = []
    for line in text.splitlines():
        b = _banner_line(line)
        if b.lower().startswith("classification:"):
            b = b.split(":", 1)[1].strip()
        if not b or len(b) > 80 or b != b.upper():
            continue
        for lv in s.levels:
            if any(re.fullmatch(p, b) for p in lv.banners):
                found.append(lv.id)
    return s.highest(found) if found else None


def _portion_re(s: Scale) -> list[tuple[str, re.Pattern]]:
    out = []
    for lv in s.levels:
        for p in lv.portions:
            out.append((lv.id, re.compile(r"^\s*\((" + p + r")\)\s")))
    return out


def detect_portion(s: Scale, text: str) -> str | None:
    """The highest portion mark opening any paragraph of a passage: "(S) ...", "(U) ...",
    "(TS//NF) ...". None when it has none."""
    pats = _portion_re(s)
    if not pats:
        return None
    found = []
    for para in re.split(r"\n\s*\n|\n(?=\s*\()", text):
        for level_id, rx in pats:
            if rx.match(para):
                found.append(level_id)
                break
    return s.highest(found) if found else None


# ----------------------------------------------------------------- output marking

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _level_of(s: Scale, text: str, source_levels: dict[int, str], context_level: str) -> str:
    cited = [source_levels.get(int(n)) for m in _CITE.finditer(text) for n in m.group(1).split(",")]
    cited = [c for c in cited if c]
    if cited:
        return s.highest(cited)
    return context_level if s.mode == "strict" else s.default


def mark_paragraphs(
    s: Scale, paragraphs: list[str], source_levels: dict[int, str], context_level: str
) -> list[str]:
    """Each paragraph's level: the highest among the sources it cites; a paragraph that
    cites nothing takes the highest level the model saw (strict) or the default (cited)."""
    return [_level_of(s, p, source_levels, context_level) for p in paragraphs]


def split_paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text or "") if p.strip()]


_ITEM = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$")


def marked_markdown(
    s: Scale, text: str, source_levels: dict[int, str], context_level: str
) -> tuple[str, str]:
    """(markdown with a portion mark before each paragraph and each list item, banner
    level). Headings, tables and code are left as they are."""
    out, levels = [], []
    for p in split_paragraphs(text):
        stripped = p.lstrip()
        if stripped.startswith(("#", "|", "```")):
            out.append(p)
            continue
        lines = p.splitlines()
        if all(
            _ITEM.match(ln) or ln.startswith((" ", "\t")) for ln in lines if ln.strip()
        ) and _ITEM.match(lines[0]):
            marked = []
            for ln in lines:
                m = _ITEM.match(ln)
                if m:
                    lv = _level_of(s, m.group(2), source_levels, context_level)
                    levels.append(lv)
                    marked.append(f"{m.group(1)}({s.level(lv).short}) {m.group(2)}")
                else:
                    marked.append(ln)
            out.append("\n".join(marked))
            continue
        lv = _level_of(s, p, source_levels, context_level)
        levels.append(lv)
        mark = f"({s.level(lv).short}) "
        out.append(stripped if stripped.startswith(">") else mark + stripped)
    return "\n\n".join(out), s.highest(levels) if levels else s.default


# ----------------------------------------------------------------- the library's levels

# A cartridge's clearance (set when it was made) read onto the company scale.
CARTRIDGE_CLEARANCE = {
    "open": "public",
    "internal": "internal",
    "confidential": "confidential",
    "restricted": "restricted",
}


async def classify_document(db, doc, s: Scale | None = None) -> tuple[str | None, int]:
    """Read one volume's markings: its level from banner lines (unless set by hand), and
    each passage's portion mark. Falls back to the cartridge it came in. Returns (the
    volume's level or None, how many passages carry a portion mark)."""
    from sqlalchemy import select, text

    from library_agent.db.models import Chunk

    s = s or load()
    chunks = list(
        (
            await db.execute(
                select(Chunk).where(Chunk.document_id == doc.id).order_by(Chunk.order_index)
            )
        ).scalars()
    )
    portions = 0
    for c in chunks:
        c.portion = detect_portion(s, c.text) if c.kind == "text" else None
        portions += c.portion is not None
    ids = {lv.id for lv in s.levels}
    if doc.classification_source == "manual" and doc.classification in ids:
        return doc.classification, portions
    level = detect_document(s, "\n".join(c.text for c in chunks if c.kind == "text"))
    marked = [c.portion for c in chunks if c.portion]
    if marked:  # a portion-marked document is at least as high as its highest portion
        level = s.highest([level, *marked]) if level else s.highest(marked)
    if level:
        doc.classification, doc.classification_source = level, "marking"
        return level, portions
    clearance = (
        (
            await db.execute(
                text(
                    "select c.design->>'clearance' from cartridge_document cd"
                    " join cartridge c on c.id = cd.cartridge_id where cd.document_id = :d"
                ),
                {"d": doc.id},
            )
        )
        .scalars()
        .all()
    )
    mapped = [CARTRIDGE_CLEARANCE.get(x or "") for x in clearance]
    mapped = [m for m in mapped if m in ids]
    if mapped:
        doc.classification, doc.classification_source = s.highest(mapped), "cartridge"
    else:
        doc.classification, doc.classification_source = None, None
    return doc.classification, portions


async def classify_all(session_scope) -> dict:
    """Every volume, one transaction each. Counts by level and passages portion-marked."""
    from collections import Counter

    from sqlalchemy import select

    from library_agent.db.models import Document

    s = load()
    async with session_scope() as db:
        ids = list((await db.execute(select(Document.id))).scalars())
    levels, portions = Counter(), 0
    for did in ids:
        async with session_scope() as db:
            doc = await db.get(Document, did)
            if doc:
                lv, n = await classify_document(db, doc, s)
                levels[lv or f"{s.default} (default)"] += 1
                portions += n
    return {"volumes": dict(levels), "portion_marked_passages": portions}


def model_limit(model: str | None, s: Scale | None = None) -> str | None:
    """The highest level a model may be shown: none for a local one; for a provider, the
    ceiling set for it in Settings › Providers, else the scale's remote ceiling."""
    from library_agent.llm import providers

    if not model:
        return None
    prov, _ = providers.split(model)
    if prov is None:
        return None
    s = s or load()
    return prov.ceiling if prov.ceiling in {lv.id for lv in s.levels} else s.remote_ceiling


def effective_ceiling(
    ceiling: str | None,
    *,
    remote: bool = False,
    models: list[str] | None = None,
    s: Scale | None = None,
) -> str | None:
    """The lowest of: a question's own ceiling; the remote ceiling, when what is retrieved
    leaves the machine some other way (the token API); and each model's own limit. None
    when nothing restricts it."""
    s = s or load()
    limits = [ceiling, s.remote_ceiling if remote else None]
    limits += [model_limit(m, s) for m in models or []]
    limits = [c for c in limits if c]
    return min(limits, key=s.rank) if limits else None


def allowed_levels(
    ceiling: str | None,
    *,
    remote: bool = False,
    models: list[str] | None = None,
    s: Scale | None = None,
) -> list[str] | None:
    """The level ids a question may draw on, or None when every level is allowed."""
    s = s or load()
    top = effective_ceiling(ceiling, remote=remote, models=models, s=s)
    if top is None:
        return None
    allowed = s.at_or_below(top)
    return None if len(allowed) == len(s.levels) else allowed


def default_level() -> str:
    """The level of a volume nobody has marked."""
    return load().default


async def hit_levels(db, hits, s: Scale | None = None) -> list[str]:
    """Each hit's level, in order: a passage its portion mark, else its volume's level,
    else the default; a reading the highest of its volume and the passages it spans; a
    live module result the level set for its module in Customize (see module_level)."""
    from sqlalchemy import text

    s = s or load()
    ids = {lv.id for lv in s.levels}
    chunk_ids = {
        str(c)
        for h in hits
        if getattr(h, "kind", "passage") != "live"
        for c in (getattr(h, "span_chunk_ids", None) or [h.chunk_id])
        if c
    }
    rows = (
        (
            await db.execute(
                text(
                    "select c.id, c.portion, d.classification from chunk c"
                    " join document d on d.id = c.document_id where c.id = any(cast(:ids as uuid[]))"
                ),
                {"ids": list(chunk_ids)},
            )
        ).all()
        if chunk_ids
        else []
    )
    doc_level = {str(i): (dl if dl in ids else None) for i, _, dl in rows}
    portion = {str(i): (p if p in ids else None) for i, p, _ in rows}
    out = []
    for h in hits:
        if getattr(h, "kind", "passage") == "live":
            # A module's results count as the level set for it in Customize.
            out.append(module_level((getattr(h, "live", None) or {}).get("module_id"), s))
            continue
        span = [str(c) for c in (getattr(h, "span_chunk_ids", None) or [h.chunk_id])]
        if getattr(h, "kind", "passage") == "reading":
            out.append(
                s.highest(
                    [doc_level.get(c) or s.default for c in span] + [portion.get(c) for c in span]
                )
            )
        else:
            c = span[0]
            out.append(portion.get(c) or doc_level.get(c) or s.default)
    return out


async def within(db, document_ids: list, ceiling: str, s: Scale | None = None) -> tuple[list, list]:
    """(volumes at or below the ceiling, volumes above it), order kept. A volume counts at
    its highest: its own level or any passage's portion mark, whichever is higher."""
    from sqlalchemy import text

    s = s or load()
    if not document_ids:
        return [], []
    rows = (
        await db.execute(
            text(
                "select d.id, d.classification,"
                " (select array_agg(distinct c.portion) from chunk c"
                "  where c.document_id = d.id and c.portion is not null)"
                " from document d where d.id = any(cast(:ids as uuid[]))"
            ),
            {"ids": [str(i) for i in document_ids]},
        )
    ).all()
    ids = {lv.id for lv in s.levels}
    level = {
        r[0]: s.highest(
            [r[1] if r[1] in ids else s.default, *[p for p in (r[2] or []) if p in ids]]
        )
        for r in rows
    }
    top = s.rank(ceiling)
    kept = [d for d in document_ids if s.rank(level.get(d, s.default)) <= top]
    return kept, [d for d in document_ids if s.rank(level.get(d, s.default)) > top]


def module_level(module_id: str | None, s: Scale | None = None) -> str:
    """The level a module's live results count as: set per module in Customize, else the
    scale's default. A level no longer on the scale falls back to the default too."""
    from library_agent.modules import store

    s = s or load()
    lv = store.config_for(module_id).classification if module_id else ""
    return lv if lv in {x.id for x in s.levels} else s.default
