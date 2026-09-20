"""Cartridges: a portable slice of a library.

A zip, not a database dump -- a dump is tied to a schema version and cannot be selective.
The receiving library inserts it onto its rack as a distinct, coloured collection and
then reads across everything it holds: threads and disagreements form between the
cartridge and the local shelf, and every margin note keeps the colour of where it came
from.

Layout::

    cartridge.json          manifest (see `Manifest`)
    icon.svg
    documents/<hash>.<ext>  originals -- `full` level only
    data/*.jsonl            documents, sections, chunks, artifacts, categories,
                            document_categories, chunk_categories, citations
    vectors/<kind>.ids.json + <kind>.f16.npy

Confidentiality is a dial on what *leaves*:

    full        originals, passages, readings     -- the receiver can open the PDF
    readings    readings only                     -- searchable, citable; never the text
    catalogue   document summaries and subjects   -- knows the material exists

At `readings` every section becomes one synthetic chunk whose text is that section's
summary, and reflections are re-targeted at it. Nothing downstream -- retrieval, the
citation apparatus, the reader -- has to know: the passage *is* the reading. The synthetic
id is derived from the content hash and section order, so it is the same on every machine.

Clusters and contradictions are never shipped: they are corpus-derived, and the receiver
must recompute them across everything it holds. That is the point."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import (
    CHUNK_NAMESPACE,
    Artifact,
    ArtifactKind,
    Cartridge,
    CartridgeDocument,
    CartridgeLevel,
    Category,
    Chunk,
    ChunkCategory,
    Citation,
    Document,
    DocumentCategory,
    DocumentStatus,
    Embedding,
    OwnerKind,
    Section,
    TargetKind,
)
from library_agent.db.purge import delete_document
from library_agent.ingest.dedup import find_exact
from library_agent.library import taxonomy
from library_agent.library.cartridge_design import (
    clamp_design,
    constellation_points,
    render_constellation,
    sanitize_art,
    store_art,
)
from library_agent.llm.embed import embed_texts
from library_agent.llm.ollama import Ollama

log = logging.getLogger(__name__)

FORMAT_VERSION = 1

# A fixed palette: these sit beside rubric, verdigris, amber and violet on the page and
# must read as *identity*, never as one of those signals.
PALETTE = [
    "#2f6f8f",  # slate blue
    "#7a5c2e",  # umber
    "#4f7a3a",  # moss
    "#8a3d5e",  # mulberry
    "#3e6b6b",  # teal-grey
    "#9a6b1f",  # ochre
    "#5a4b8a",  # dusk
    "#6f6f6f",  # graphite
]

DEFAULT_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.5"><rect x="4" y="3" width="16" height="18" rx="2"/>'
    '<path d="M8 7h8M8 11h8M8 15h5"/></svg>'
)

DATA_FILES = (
    "documents",
    "sections",
    "chunks",
    "artifacts",
    "categories",
    "document_categories",
    "chunk_categories",
    "citations",
)
VECTOR_KINDS = ("chunk", "artifact", "document")

_CATALOGUE_ARTIFACTS = {ArtifactKind.ORIENTATION, ArtifactKind.DOCUMENT_SUMMARY}
_HASH = re.compile(r"[0-9a-f]{64}")
_ORIGINAL_EXTS = {".pdf", ".md", ".markdown", ".txt", ".text", ".rst"}


_SVG_TAGS = {
    "svg",
    "g",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "title",
}
_SVG_ATTRS = {
    "viewBox",
    "width",
    "height",
    "fill",
    "stroke",
    "stroke-width",
    "stroke-linecap",
    "stroke-linejoin",
    "fill-rule",
    "opacity",
    "d",
    "x",
    "y",
    "rx",
    "ry",
    "cx",
    "cy",
    "r",
    "x1",
    "y1",
    "x2",
    "y2",
    "points",
    "transform",
}


def sanitize_svg(svg: str | None) -> str:
    """An icon arrives from someone else's machine and lands in innerHTML. Keep only plain
    shapes and their geometry -- no scripts, handlers, links, styles, images or foreign
    objects -- and re-serialise from the tree so nothing rides along in the text."""
    import xml.etree.ElementTree as ET

    if not svg or len(svg) > 20_000:
        return DEFAULT_ICON
    try:
        root = ET.fromstring(svg.strip())
    except ET.ParseError:
        return DEFAULT_ICON
    ns = "{http://www.w3.org/2000/svg}"

    def clean(el: ET.Element) -> ET.Element | None:
        tag = el.tag.removeprefix(ns)
        if tag not in _SVG_TAGS:
            return None
        out = ET.Element(tag)
        for k, v in el.attrib.items():
            if k in _SVG_ATTRS and "url(" not in v and "javascript" not in v.lower():
                out.set(k, v)
        if tag == "title":
            out.text = (el.text or "")[:80]
        for child in el:
            c = clean(child)
            if c is not None:
                out.append(c)
        return out

    top = clean(root)
    if top is None or top.tag != "svg":
        return DEFAULT_ICON
    top.set("xmlns", "http://www.w3.org/2000/svg")
    return ET.tostring(top, encoding="unicode")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return (s or "cartridge")[:60]


def reading_chunk_id(content_hash: str, order_index: int) -> uuid.UUID:
    """The synthetic chunk standing in for a section at the `readings` level. Keyed on
    section *order*, not section id -- section ids are random per machine, order is not."""
    return uuid.uuid5(CHUNK_NAMESPACE, f"{content_hash}:reading:{order_index}")


class CartridgeError(ValueError):
    pass


# --------------------------------------------------------------------------- selection


async def resolve_selection(
    db: AsyncSession,
    *,
    document_ids: list[uuid.UUID] | None = None,
    category_ids: list[uuid.UUID] | None = None,
    cartridge_ids: list[uuid.UUID] | None = None,
) -> list[uuid.UUID]:
    """Union of explicit documents, everything under the given subjects, and everything in
    the given cartridges. Order is stable (by title) so previews and exports agree."""
    ids: set[uuid.UUID] = set(document_ids or [])
    if category_ids:
        ids |= set(
            (
                await db.execute(
                    select(DocumentCategory.document_id).where(
                        DocumentCategory.category_id.in_(category_ids)
                    )
                )
            ).scalars()
        )
    if cartridge_ids:
        ids |= set(
            (
                await db.execute(
                    select(CartridgeDocument.document_id).where(
                        CartridgeDocument.cartridge_id.in_(cartridge_ids)
                    )
                )
            ).scalars()
        )
    if not ids:
        return []
    # A volume that arrived only in a restricted cartridge does not leave again.
    restricted = set(
        (
            await db.execute(
                select(CartridgeDocument.document_id)
                .join(Cartridge, Cartridge.id == CartridgeDocument.cartridge_id)
                .where(
                    CartridgeDocument.document_id.in_(ids),
                    CartridgeDocument.introduced.is_(True),
                    Cartridge.design["clearance"].astext == "restricted",
                )
            )
        ).scalars()
    )
    rows = (
        await db.execute(
            select(Document.id)
            .where(Document.id.in_(ids - restricted), Document.status == DocumentStatus.READY)
            .order_by(Document.title)
        )
    ).scalars()
    return list(rows)


# --------------------------------------------------------------------------- export


@dataclass
class Bundle:
    """Everything gathered for one export, already reduced to the requested level."""

    level: str
    documents: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    categories: list[dict] = field(default_factory=list)
    document_categories: list[dict] = field(default_factory=list)
    chunk_categories: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    # owner id -> vector, per kind
    vectors: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {k: {} for k in VECTOR_KINDS}
    )
    originals: list[tuple[str, Path]] = field(default_factory=list)  # (arcname, path)

    def counts(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "sections": len(self.sections),
            "chunks": len(self.chunks),
            "artifacts": len(self.artifacts),
            "reflections": sum(1 for a in self.artifacts if a["kind"] == ArtifactKind.REFLECTION),
            "categories": len(self.categories),
            "citations": len(self.citations),
            "vectors": sum(len(v) for v in self.vectors.values()),
            "originals": len(self.originals),
        }

    def estimated_bytes(self) -> int:
        texty = sum(len(json.dumps(r, default=str)) for r in self.chunks + self.artifacts)
        vec = sum(len(v) for v in self.vectors.values()) * settings().embed_dim * 2
        files = sum(p.stat().st_size for _, p in self.originals if p.exists())
        # jsonl compresses ~4x; float16 vectors and PDFs barely at all.
        return texty // 4 + vec + files


def _j(v: Any) -> Any:
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def _row(obj: Any, cols: tuple[str, ...]) -> dict:
    return {c: _j(getattr(obj, c)) for c in cols}


_DOC_COLS = (
    "id",
    "content_hash",
    "title",
    "authors",
    "kind",
    "original_filename",
    "page_count",
    "doi",
    "tier",
    "added_at",
)
_SEC_COLS = (
    "id",
    "document_id",
    "parent_id",
    "order_index",
    "path",
    "title",
    "level",
    "char_start",
    "char_end",
    "page_start",
    "page_end",
)
_CHUNK_COLS = (
    "id",
    "document_id",
    "section_id",
    "order_index",
    "text",
    "context_prefix",
    "token_count",
    "page_start",
    "char_start",
    "char_end",
)
_ART_COLS = (
    "id",
    "kind",
    "target_kind",
    "target_id",
    "text",
    "data",
    "model",
    "prompt_version",
    "tier",
)


async def _vectors_for(
    db: AsyncSession, kind: str, owner_ids: list[uuid.UUID]
) -> dict[str, list[float]]:
    if not owner_ids:
        return {}
    out: dict[str, list[float]] = {}
    for i in range(0, len(owner_ids), 500):
        batch = [str(x) for x in owner_ids[i : i + 500]]
        rows = await db.execute(
            text(
                "select owner_id, cast(vec as text) as v from embedding "
                "where owner_kind = :k and model = :m and owner_id = any(cast(:ids as uuid[]))"
            ),
            {"k": kind, "m": settings().embed_model, "ids": batch},
        )
        for r in rows:
            out[str(r.owner_id)] = json.loads(r.v)
    return out


async def gather(db: AsyncSession, document_ids: list[uuid.UUID], level: str) -> Bundle:
    """Collect and reduce. This is the one place the level is interpreted on the way out."""
    level = CartridgeLevel(level)
    b = Bundle(level=level)
    if not document_ids:
        return b

    docs = list((await db.execute(select(Document).where(Document.id.in_(document_ids)))).scalars())
    doc_ids = [d.id for d in docs]
    for d in docs:
        b.documents.append(_row(d, _DOC_COLS))
        if level == CartridgeLevel.FULL and d.source_path and Path(d.source_path).exists():
            p = Path(d.source_path)
            b.originals.append((f"documents/{d.content_hash}{p.suffix.lower()}", p))
    hash_of = {d.id: d.content_hash for d in docs}

    sections = list(
        (
            await db.execute(
                select(Section)
                .where(Section.document_id.in_(doc_ids))
                .order_by(Section.order_index)
            )
        ).scalars()
    )
    if level != CartridgeLevel.CATALOGUE:
        b.sections = [_row(s, _SEC_COLS) for s in sections]
    sec_ids = [s.id for s in sections]

    chunks = list((await db.execute(select(Chunk).where(Chunk.document_id.in_(doc_ids)))).scalars())
    chunk_ids = [c.id for c in chunks]
    chunk_section = {c.id: c.section_id for c in chunks}

    # Artifacts on the document, its sections, or its chunks. Where your own reading and
    # a cartridge's both exist for the same target, yours ships; theirs only stands in
    # when you have none (a readings-only volume, say).
    arts_all = list(
        (
            await db.execute(
                select(Artifact).where(
                    Artifact.target_id.in_(doc_ids + sec_ids + chunk_ids),
                    Artifact.kind.in_(
                        [
                            ArtifactKind.ORIENTATION,
                            ArtifactKind.DOCUMENT_SUMMARY,
                            ArtifactKind.SECTION_SUMMARY,
                            ArtifactKind.REFLECTION,
                        ]
                    ),
                )
            )
        ).scalars()
    )
    arts: list[Artifact] = []
    taken: set[tuple[str, str, uuid.UUID]] = set()
    for a in sorted(arts_all, key=lambda a: a.cartridge_id is not None):
        key = (a.kind, a.target_kind, a.target_id)
        if key not in taken:
            taken.add(key)
            arts.append(a)

    # Subjects.
    dcs = list(
        (
            await db.execute(
                select(DocumentCategory).where(DocumentCategory.document_id.in_(doc_ids))
            )
        ).scalars()
    )
    ccs = (
        list(
            (
                await db.execute(select(ChunkCategory).where(ChunkCategory.chunk_id.in_(chunk_ids)))
            ).scalars()
        )
        if chunk_ids
        else []
    )
    cat_ids = {x.category_id for x in dcs} | {x.category_id for x in ccs}
    if cat_ids:
        cats = (await db.execute(select(Category).where(Category.id.in_(cat_ids)))).scalars()
        b.categories = [_row(c, ("id", "name", "description")) for c in cats]
    b.document_categories = [_row(x, ("document_id", "category_id", "confidence")) for x in dcs]

    # Citations within the slice; edges out of it keep the raw reference and lose the match.
    cits = (
        await db.execute(select(Citation).where(Citation.citing_document_id.in_(doc_ids)))
    ).scalars()
    for c in cits:
        b.citations.append(
            {
                "citing_document_id": str(c.citing_document_id),
                "matched_document_id": str(c.matched_document_id)
                if c.matched_document_id in hash_of
                else None,
                "raw_reference": c.raw_reference,
                "doi": c.doi,
                "confidence": c.confidence,
            }
        )

    if level == CartridgeLevel.FULL:
        b.chunks = [_row(c, _CHUNK_COLS) for c in chunks]
        b.artifacts = [_row(a, _ART_COLS) for a in arts]
        b.chunk_categories = [_row(x, ("chunk_id", "category_id", "confidence")) for x in ccs]
        b.vectors["chunk"] = await _vectors_for(db, OwnerKind.CHUNK, chunk_ids)
        b.vectors["artifact"] = await _vectors_for(db, OwnerKind.ARTIFACT, [a.id for a in arts])
        b.vectors["document"] = await _vectors_for(db, OwnerKind.DOCUMENT, doc_ids)
        return b

    if level == CartridgeLevel.CATALOGUE:
        b.artifacts = [_row(a, _ART_COLS) for a in arts if a.kind in _CATALOGUE_ARTIFACTS]
        b.vectors["artifact"] = await _vectors_for(
            db, OwnerKind.ARTIFACT, [uuid.UUID(a["id"]) for a in b.artifacts]
        )
        b.vectors["document"] = await _vectors_for(db, OwnerKind.DOCUMENT, doc_ids)
        return b

    # readings: one synthetic chunk per summarised section, reflections re-targeted.
    summaries = {a.target_id: a for a in arts if a.kind == ArtifactKind.SECTION_SUMMARY}
    synthetic: dict[uuid.UUID, uuid.UUID] = {}  # section id -> synthetic chunk id
    order_in: dict[uuid.UUID, int] = {}
    for s in sections:
        art = summaries.get(s.id)
        if not art:
            continue
        cid = reading_chunk_id(hash_of[s.document_id], s.order_index)
        synthetic[s.id] = cid
        order = order_in.get(s.document_id, 0)
        order_in[s.document_id] = order + 1
        b.chunks.append(
            {
                "id": str(cid),
                "document_id": str(s.document_id),
                "section_id": str(s.id),
                "order_index": order,
                "text": art.text,
                "context_prefix": "",
                "token_count": max(1, len(art.text) // 4),
                "page_start": s.page_start,
                "char_start": s.char_start,
                "char_end": s.char_end,
            }
        )
    art_vecs = await _vectors_for(db, OwnerKind.ARTIFACT, [a.id for a in arts])
    for a in arts:
        if a.kind == ArtifactKind.REFLECTION:
            sec = chunk_section.get(a.target_id)
            target = synthetic.get(sec) if sec else None
            if not target:
                continue
            row = _row(a, _ART_COLS)
            row["target_id"] = str(target)
            b.artifacts.append(row)
        else:
            b.artifacts.append(_row(a, _ART_COLS))
        if str(a.id) in art_vecs:
            b.vectors["artifact"][str(a.id)] = art_vecs[str(a.id)]
    # The summary's own vector doubles as the synthetic chunk's: same text, same model.
    for sec_id, cid in synthetic.items():
        v = art_vecs.get(str(summaries[sec_id].id))
        if v:
            b.vectors["chunk"][str(cid)] = v
    # Chunk subjects roll up to the section's synthetic chunk.
    seen: set[tuple[str, str]] = set()
    for x in ccs:
        sec = chunk_section.get(x.chunk_id)
        target = synthetic.get(sec) if sec else None
        if target and (key := (str(target), str(x.category_id))) not in seen:
            seen.add(key)
            b.chunk_categories.append(
                {"chunk_id": key[0], "category_id": key[1], "confidence": x.confidence}
            )
    b.vectors["document"] = await _vectors_for(db, OwnerKind.DOCUMENT, doc_ids)
    return b


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows).encode()


def _pack_vectors(vecs: dict[str, list[float]]) -> tuple[bytes, bytes]:
    ids = list(vecs)
    arr = (
        np.asarray([vecs[i] for i in ids], dtype=np.float16)
        if ids
        else np.zeros((0, 0), np.float16)
    )
    buf = io.BytesIO()
    np.save(buf, arr)
    return json.dumps(ids).encode(), buf.getvalue()


def _digest(parts: list[tuple[str, bytes]]) -> str:
    h = hashlib.sha256()
    for name, data in sorted(parts):
        h.update(name.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


async def build_cartridge(
    db: AsyncSession,
    *,
    document_ids: list[uuid.UUID],
    level: str,
    name: str,
    colour: str | None = None,
    icon_svg: str | None = None,
    made_by: str | None = None,
    cartridge_id: uuid.UUID | None = None,
    version: int = 1,
    out_dir: Path | None = None,
    design: dict | None = None,
    art_png: bytes | None = None,
) -> Path:
    """Write the zip and return its path. `cartridge_id` is stable across versions of the
    same cartridge; a fresh export gets a fresh id."""
    if not document_ids:
        raise CartridgeError("nothing selected")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", colour or PALETTE[0]):
        raise CartridgeError("colour must be #rrggbb")
    cfg = settings()
    b = await gather(db, document_ids, level)
    cid = cartridge_id or uuid.uuid4()
    slug = slugify(name)

    parts: list[tuple[str, bytes]] = []
    for key in DATA_FILES:
        parts.append((f"data/{key}.jsonl", _jsonl(getattr(b, key))))
    for kind in VECTOR_KINDS:
        ids, arr = _pack_vectors(b.vectors[kind])
        parts.append((f"vectors/{kind}.ids.json", ids))
        parts.append((f"vectors/{kind}.f16.npy", arr))
    for arcname, path in b.originals:
        parts.append((arcname, path.read_bytes()))
    # The label: the maker's upload, or the cartridge's own constellation.
    design = clamp_design(design)
    if art_png is None:
        pts = await constellation_points(db, document_ids)
        art_png = render_constellation(pts, colour or PALETTE[0])
        design["art"] = "generated"
    parts.append(("art/label.png", art_png))

    manifest = {
        "format_version": FORMAT_VERSION,
        "id": str(cid),
        "name": name.strip() or slug,
        "slug": slug,
        "version": version,
        "colour": colour or PALETTE[0],
        "made_by": made_by,
        "made_at": datetime.now(UTC).isoformat(),
        "level": str(CartridgeLevel(level)),
        "embed_model": cfg.embed_model,
        "embed_dim": cfg.embed_dim,
        "reader_model": cfg.reader_model,
        "prompt_versions": dict(cfg.prompt_versions),
        "counts": b.counts(),
        "design": design,
        "art": "art/label.png",
        "content_hash": _digest(parts),
        "signature": None,
    }

    out_dir = out_dir or cfg.storage_dir.parent / "cartridges"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slug}-v{version}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("cartridge.json", json.dumps(manifest, indent=2))
        z.writestr("icon.svg", sanitize_svg(icon_svg))
        for arcname, data in parts:
            z.writestr(
                arcname,
                data,
                compress_type=zipfile.ZIP_STORED
                if arcname.startswith("documents/") or arcname.endswith(".npy")
                else zipfile.ZIP_DEFLATED,
            )
    return out


# --------------------------------------------------------------------------- import


def read_manifest(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        if "cartridge.json" not in z.namelist():
            raise CartridgeError("not a cartridge: no cartridge.json")
        m = json.loads(z.read("cartridge.json"))
    fv = m.get("format_version")
    if not isinstance(fv, int) or fv > FORMAT_VERSION:
        raise CartridgeError(
            f"cartridge format {fv} is newer than this library understands ({FORMAT_VERSION})"
        )
    for k in ("id", "name", "level", "embed_model", "content_hash"):
        if k not in m:
            raise CartridgeError(f"manifest missing {k}")
    CartridgeLevel(m["level"])
    return m


def is_cartridge(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return "cartridge.json" in z.namelist()
    except (zipfile.BadZipFile, OSError):
        return False


@dataclass
class ImportResult:
    cartridge_id: uuid.UUID
    name: str
    version: int
    documents_introduced: int = 0
    documents_joined: int = 0  # already on the shelf; membership only
    artifacts: int = 0
    artifacts_skipped: int = 0
    vectors_loaded: int = 0
    vectors_embedded: int = 0
    replaced_version: int | None = None
    noop: bool = False


def _load(z: zipfile.ZipFile, name: str) -> list[dict]:
    if name not in z.namelist():
        return []
    return [json.loads(line) for line in z.read(name).decode().splitlines() if line.strip()]


def _load_vectors(z: zipfile.ZipFile, kind: str) -> dict[str, np.ndarray]:
    ids_name, arr_name = f"vectors/{kind}.ids.json", f"vectors/{kind}.f16.npy"
    if ids_name not in z.namelist() or arr_name not in z.namelist():
        return {}
    ids = json.loads(z.read(ids_name))
    arr = np.load(io.BytesIO(z.read(arr_name)), allow_pickle=False)
    return {i: arr[n] for n, i in enumerate(ids)} if len(ids) else {}


def _verify(z: zipfile.ZipFile, manifest: dict) -> None:
    parts = [
        (n, z.read(n))
        for n in z.namelist()
        if n not in ("cartridge.json", "icon.svg") and not n.endswith("/")
    ]
    if _digest(parts) != manifest["content_hash"]:
        raise CartridgeError("content hash does not match; the cartridge was altered or damaged")


def _u(v: str | None) -> uuid.UUID | None:
    return uuid.UUID(v) if v else None


async def _remove_cartridge_artifacts(db: AsyncSession, cartridge_id: uuid.UUID) -> None:
    ids = list(
        (
            await db.execute(select(Artifact.id).where(Artifact.cartridge_id == cartridge_id))
        ).scalars()
    )
    if ids:
        await db.execute(
            delete(Embedding).where(
                Embedding.owner_kind == OwnerKind.ARTIFACT, Embedding.owner_id.in_(ids)
            )
        )
        await db.execute(delete(Artifact).where(Artifact.id.in_(ids)))


async def import_cartridge(
    db: AsyncSession, zip_path: Path, *, client: Ollama | None = None
) -> ImportResult:
    cfg = settings()
    manifest = read_manifest(zip_path)
    cid = uuid.UUID(manifest["id"])
    level = CartridgeLevel(manifest["level"])
    version = int(manifest.get("version", 1))
    res = ImportResult(cartridge_id=cid, name=manifest["name"], version=version)

    existing = await db.get(Cartridge, cid)
    if existing and existing.version >= version:
        res.noop = True
        return res

    with zipfile.ZipFile(zip_path) as z:
        _verify(z, manifest)
        icon = sanitize_svg(
            z.read("icon.svg").decode(errors="replace") if "icon.svg" in z.namelist() else None
        )
        docs = _load(z, "data/documents.jsonl")
        sections = _load(z, "data/sections.jsonl")
        chunks = _load(z, "data/chunks.jsonl")
        artifacts = _load(z, "data/artifacts.jsonl")
        categories = _load(z, "data/categories.jsonl")
        doc_cats = _load(z, "data/document_categories.jsonl")
        chunk_cats = _load(z, "data/chunk_categories.jsonl")
        citations = _load(z, "data/citations.jsonl")
        vectors = {k: _load_vectors(z, k) for k in VECTOR_KINDS}
        originals = {
            n.split("/", 1)[1]: n
            for n in z.namelist()
            if n.startswith("documents/") and not n.endswith("/")
        }

        incoming_hashes = {d["content_hash"] for d in docs}
        if existing:
            # Upgrade: drop what the old version wrote, keep documents the new version
            # still carries (and every local note on them), remove the rest.
            res.replaced_version = existing.version
            await _remove_cartridge_artifacts(db, cid)
            old = list(
                (
                    await db.execute(
                        select(CartridgeDocument, Document.content_hash)
                        .join(Document, Document.id == CartridgeDocument.document_id)
                        .where(CartridgeDocument.cartridge_id == cid)
                    )
                ).all()
            )
            for m, h in old:
                await db.execute(
                    delete(CartridgeDocument).where(
                        CartridgeDocument.cartridge_id == cid,
                        CartridgeDocument.document_id == m.document_id,
                    )
                )
                gone = m.introduced and h not in incoming_hashes
                if gone and not await _has_other_membership(db, m.document_id, cid):
                    await delete_document(db, m.document_id)
            await db.flush()

        row = existing or Cartridge(id=cid)
        row.name = manifest["name"]
        row.slug = manifest.get("slug") or slugify(manifest["name"])
        row.version = version
        row.colour = manifest.get("colour") or PALETTE[0]
        row.icon_svg = icon
        row.made_by = manifest.get("made_by")
        row.made_at = (
            datetime.fromisoformat(manifest["made_at"]) if manifest.get("made_at") else None
        )
        row.level = level
        row.embed_model = manifest["embed_model"]
        row.reader_model = manifest.get("reader_model")
        row.prompt_versions = manifest.get("prompt_versions")
        row.manifest = manifest
        row.content_hash = manifest["content_hash"]
        row.document_count = len(docs)
        row.design = clamp_design(manifest.get("design"))
        art_name = manifest.get("art") or "art/label.png"
        if art_name in z.namelist():
            try:
                row.art_path = str(store_art(cid, sanitize_art(z.read(art_name))))
            except ValueError:
                row.art_path = None  # bad art is dropped, not fatal
        if not existing:
            db.add(row)
        await db.flush()

        same_model = (
            manifest["embed_model"] == cfg.embed_model
            and int(manifest.get("embed_dim") or cfg.embed_dim) == cfg.embed_dim
        )

        # id maps: cartridge id -> local id
        doc_map: dict[str, uuid.UUID] = {}
        sec_map: dict[str, uuid.UUID] = {}
        chunk_map: dict[str, uuid.UUID] = {}
        introduced: set[uuid.UUID] = set()
        sections_by_doc: dict[str, list[dict]] = {}
        for s in sections:
            sections_by_doc.setdefault(s["document_id"], []).append(s)
        chunks_by_doc: dict[str, list[dict]] = {}
        for c in chunks:
            chunks_by_doc.setdefault(c["document_id"], []).append(c)

        to_embed_chunks: list[tuple[uuid.UUID, str]] = []
        to_embed_docs: list[tuple[uuid.UUID, str]] = []

        for d in docs:
            if not _HASH.fullmatch(str(d.get("content_hash", ""))):
                raise CartridgeError("document with a malformed content hash")
            local = await find_exact(db, d["content_hash"])
            if (
                local
                and local.readings_only
                and level == CartridgeLevel.FULL
                and not await _has_other_membership(db, local.id, cid)
            ):
                # We only had their reading of it; now we have the thing itself.
                await delete_document(db, local.id)
                local = None
            if local:
                doc_map[d["id"]] = local.id
                res.documents_joined += 1
                # Map their sections to ours by order; same bytes, same structure.
                ours = {
                    s.order_index: s.id
                    for s in (
                        await db.execute(select(Section).where(Section.document_id == local.id))
                    ).scalars()
                }
                for s in sections_by_doc.get(d["id"], []):
                    if s["order_index"] in ours:
                        sec_map[s["id"]] = ours[s["order_index"]]
                have = set(
                    (
                        await db.execute(select(Chunk.id).where(Chunk.document_id == local.id))
                    ).scalars()
                )
                for c in chunks_by_doc.get(d["id"], []):
                    if uuid.UUID(c["id"]) in have:
                        chunk_map[c["id"]] = uuid.UUID(c["id"])
                await db.merge(
                    CartridgeDocument(cartridge_id=cid, document_id=local.id, introduced=False)
                )
                continue

            # New to this shelf.
            new_id = uuid.uuid4()
            source_path = ""
            if level == CartridgeLevel.FULL:
                stored = next(
                    (n for h, n in originals.items() if h.startswith(d["content_hash"])), None
                )
                if stored:
                    # Names inside the zip are untrusted: the hash must be a hash, the
                    # extension one we read, and the destination inside the store.
                    ext = Path(stored).suffix.lower()
                    if ext not in _ORIGINAL_EXTS:
                        raise CartridgeError(f"original with unexpected type {ext!r}")
                    dest = cfg.storage_dir / d["content_hash"][:2] / f"{d['content_hash']}{ext}"
                    if not dest.resolve().is_relative_to(cfg.storage_dir.resolve()):
                        raise CartridgeError("original would land outside the store")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        dest.write_bytes(z.read(stored))
                    source_path = str(dest)
            doc = Document(
                id=new_id,
                content_hash=d["content_hash"],
                title=d["title"],
                authors=d.get("authors"),
                kind=d.get("kind") or "paper",
                source_path=source_path,
                original_filename=d.get("original_filename") or d["title"],
                page_count=d.get("page_count"),
                doi=d.get("doi"),
                tier=int(d.get("tier") or 0),
                status=DocumentStatus.READY,
                readings_only=level != CartridgeLevel.FULL,
            )
            db.add(doc)
            await db.flush()
            doc_map[d["id"]] = new_id
            introduced.add(new_id)
            res.documents_introduced += 1
            db.add(CartridgeDocument(cartridge_id=cid, document_id=new_id, introduced=True))

            # Sections first (parents before children: the export ordered them).
            for s in sections_by_doc.get(d["id"], []):
                sid = uuid.uuid4()
                sec_map[s["id"]] = sid
            for s in sections_by_doc.get(d["id"], []):
                db.add(
                    Section(
                        id=sec_map[s["id"]],
                        document_id=new_id,
                        parent_id=sec_map.get(s["parent_id"]) if s.get("parent_id") else None,
                        order_index=s["order_index"],
                        path=s.get("path") or "",
                        title=s.get("title"),
                        level=s.get("level") or 1,
                        char_start=s.get("char_start") or 0,
                        char_end=s.get("char_end") or 0,
                        page_start=s.get("page_start"),
                        page_end=s.get("page_end"),
                    )
                )
            # No relationships are declared between these tables, so the unit of work
            # will not order the inserts for us.
            await db.flush()
            for c in chunks_by_doc.get(d["id"], []):
                chid = uuid.UUID(c["id"])
                chunk_map[c["id"]] = chid
                db.add(
                    Chunk(
                        id=chid,
                        document_id=new_id,
                        section_id=sec_map.get(c["section_id"]) if c.get("section_id") else None,
                        order_index=c["order_index"],
                        text=c["text"],
                        context_prefix=c.get("context_prefix") or "",
                        token_count=c.get("token_count") or 0,
                        page_start=c.get("page_start"),
                        char_start=c.get("char_start") or 0,
                        char_end=c.get("char_end") or 0,
                    )
                )
                v = vectors["chunk"].get(c["id"])
                if same_model and v is not None:
                    db.add(
                        Embedding(
                            owner_kind=OwnerKind.CHUNK,
                            owner_id=chid,
                            model=cfg.embed_model,
                            vec=v.astype(np.float32).tolist(),
                        )
                    )
                    res.vectors_loaded += 1
                else:
                    to_embed_chunks.append(
                        (chid, f"{c.get('context_prefix') or ''}\n\n{c['text']}".strip())
                    )
            v = vectors["document"].get(d["id"])
            if same_model and v is not None:
                db.add(
                    Embedding(
                        owner_kind=OwnerKind.DOCUMENT,
                        owner_id=new_id,
                        model=cfg.embed_model,
                        vec=v.astype(np.float32).tolist(),
                    )
                )
                res.vectors_loaded += 1
            else:
                to_embed_docs.append((new_id, d["title"]))
        await db.flush()

        # Artifacts: authored by the cartridge. Targets that did not resolve (their
        # synthetic reading-chunks on a document we hold in full) are dropped.
        to_embed_arts: list[tuple[uuid.UUID, str]] = []
        title_of = {d["id"]: d["title"] for d in docs}
        doc_of_section = {s["id"]: s["document_id"] for s in sections}
        doc_of_chunk = {c["id"]: c["document_id"] for c in chunks}
        art_ids_seen: set[tuple[str, str, uuid.UUID]] = set()
        for a in artifacts:
            tk = a["target_kind"]
            target = {
                TargetKind.DOCUMENT: doc_map,
                TargetKind.SECTION: sec_map,
                TargetKind.CHUNK: chunk_map,
            }.get(tk, {}).get(a["target_id"])
            if not target or (a["kind"], tk, target) in art_ids_seen:
                res.artifacts_skipped += 1
                continue
            art_ids_seen.add((a["kind"], tk, target))
            aid = uuid.uuid4()
            db.add(
                Artifact(
                    id=aid,
                    kind=a["kind"],
                    target_kind=tk,
                    target_id=target,
                    text=a["text"],
                    data=a.get("data"),
                    model=a.get("model") or manifest.get("reader_model") or "unknown",
                    prompt_version=a.get("prompt_version") or "v1",
                    tier=a.get("tier") or 1,
                    cartridge_id=cid,
                )
            )
            res.artifacts += 1
            v = vectors["artifact"].get(a["id"])
            if same_model and v is not None:
                db.add(
                    Embedding(
                        owner_kind=OwnerKind.ARTIFACT,
                        owner_id=aid,
                        model=cfg.embed_model,
                        vec=v.astype(np.float32).tolist(),
                    )
                )
                res.vectors_loaded += 1
            elif a["kind"] in (ArtifactKind.DOCUMENT_SUMMARY, ArtifactKind.SECTION_SUMMARY):
                src_doc = (
                    a["target_id"]
                    if tk == TargetKind.DOCUMENT
                    else doc_of_section.get(a["target_id"], "")
                )
                to_embed_arts.append((aid, f"{title_of.get(src_doc, '')}\n\n{a['text']}"))
        await db.flush()

        # Subjects merge by name.
        cat_map: dict[str, uuid.UUID] = {}
        for c in categories:
            cat = await taxonomy.get_or_create(db, c["name"])
            if cat:
                cat_map[c["id"]] = cat.id
        for x in doc_cats:
            did, catid = doc_map.get(x["document_id"]), cat_map.get(x["category_id"])
            if did and catid and did in introduced:
                await db.merge(
                    DocumentCategory(
                        document_id=did, category_id=catid, confidence=x.get("confidence", 1.0)
                    )
                )
        for x in chunk_cats:
            chid, catid = chunk_map.get(x["chunk_id"]), cat_map.get(x["category_id"])
            if chid and catid and doc_map.get(doc_of_chunk.get(x["chunk_id"], "")) in introduced:
                await db.merge(
                    ChunkCategory(
                        chunk_id=chid, category_id=catid, confidence=x.get("confidence", 1.0)
                    )
                )
        for x in citations:
            citing = doc_map.get(x["citing_document_id"])
            if citing and citing in introduced:
                db.add(
                    Citation(
                        citing_document_id=citing,
                        matched_document_id=doc_map.get(x["matched_document_id"])
                        if x.get("matched_document_id")
                        else None,
                        raw_reference=x["raw_reference"],
                        doi=x.get("doi"),
                        confidence=x.get("confidence") or 0.0,
                    )
                )
        await db.flush()

    # Anything the shipped vectors could not cover.
    pending = [
        (OwnerKind.CHUNK, to_embed_chunks),
        (OwnerKind.ARTIFACT, to_embed_arts),
        (OwnerKind.DOCUMENT, to_embed_docs),
    ]
    if any(items for _, items in pending):
        own = client is None
        c = client or Ollama()
        try:
            for kind, items in pending:
                if not items:
                    continue
                vecs = await embed_texts([t for _, t in items], c)
                for (oid, _), vec in zip(items, vecs, strict=True):
                    db.add(Embedding(owner_kind=kind, owner_id=oid, model=cfg.embed_model, vec=vec))
                    res.vectors_embedded += 1
        finally:
            if own:
                await c.aclose()
    await db.flush()
    return res


async def _has_other_membership(
    db: AsyncSession, document_id: uuid.UUID, cartridge_id: uuid.UUID
) -> bool:
    n = (
        await db.execute(
            select(func.count()).where(
                CartridgeDocument.document_id == document_id,
                CartridgeDocument.cartridge_id != cartridge_id,
            )
        )
    ).scalar_one()
    return bool(n)


@dataclass
class EjectResult:
    documents_removed: int = 0
    documents_kept: int = 0
    artifacts_removed: int = 0


async def eject_cartridge(db: AsyncSession, cartridge_id: uuid.UUID) -> EjectResult:
    """Remove what the cartridge brought: its notes everywhere, and the documents it
    introduced that nothing else holds. Documents that were already on the shelf, or that
    another cartridge also carries, stay -- with your own notes on them intact."""
    res = EjectResult()
    cart = await db.get(Cartridge, cartridge_id)
    if not cart:
        raise CartridgeError("no such cartridge")
    res.artifacts_removed = (
        await db.execute(select(func.count()).where(Artifact.cartridge_id == cartridge_id))
    ).scalar_one()
    await _remove_cartridge_artifacts(db, cartridge_id)
    members = list(
        (
            await db.execute(
                select(CartridgeDocument).where(CartridgeDocument.cartridge_id == cartridge_id)
            )
        ).scalars()
    )
    for m in members:
        await db.execute(
            delete(CartridgeDocument).where(
                CartridgeDocument.cartridge_id == cartridge_id,
                CartridgeDocument.document_id == m.document_id,
            )
        )
        if m.introduced and not await _has_other_membership(db, m.document_id, cartridge_id):
            await delete_document(db, m.document_id)
            res.documents_removed += 1
        else:
            res.documents_kept += 1
    if cart.art_path:
        import shutil

        shutil.rmtree(Path(cart.art_path).parent, ignore_errors=True)
    await db.delete(cart)
    await db.flush()
    return res


async def list_cartridges(db: AsyncSession) -> list[dict]:
    rows = (
        await db.execute(
            select(Cartridge, func.count(CartridgeDocument.document_id))
            .outerjoin(CartridgeDocument, CartridgeDocument.cartridge_id == Cartridge.id)
            .group_by(Cartridge.id)
            .order_by(Cartridge.imported_at)
        )
    ).all()
    return [
        {
            "id": str(c.id),
            "name": c.name,
            "slug": c.slug,
            "version": c.version,
            "colour": c.colour,
            "icon_svg": c.icon_svg,
            "made_by": c.made_by,
            "made_at": c.made_at.isoformat() if c.made_at else None,
            "level": c.level,
            "embed_model": c.embed_model,
            "reader_model": c.reader_model,
            "document_count": n,
            "imported_at": c.imported_at.isoformat() if c.imported_at else None,
            "design": clamp_design(c.design),
            "has_art": bool(c.art_path and Path(c.art_path).exists()),
            # Made on this machine from a folder: its maker is here, so it may be edited.
            "editable": c.made_by == "import",
        }
        for c, n in rows
    ]


async def cartridge_provenance(
    db: AsyncSession, document_ids: list[uuid.UUID]
) -> dict[uuid.UUID, dict]:
    """document id -> {id, name, colour} of the cartridge it belongs to. A document in
    several cartridges reports the first it was inserted from; a local one is absent."""
    if not document_ids:
        return {}
    rows = (
        await db.execute(
            select(CartridgeDocument.document_id, Cartridge.id, Cartridge.name, Cartridge.colour)
            .join(Cartridge, Cartridge.id == CartridgeDocument.cartridge_id)
            .where(CartridgeDocument.document_id.in_(document_ids))
            .order_by(Cartridge.imported_at)
        )
    ).all()
    out: dict[uuid.UUID, dict] = {}
    for did, cid, name, colour in rows:
        out.setdefault(did, {"id": str(cid), "name": name, "colour": colour})
    return out
