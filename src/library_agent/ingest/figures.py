"""The figures in a PDF, found and rendered on demand.

Extraction already knows where the figures are -- `_figure_regions` finds the drawings
and raster images on a page so their labels can be kept out of the prose. This module
uses the same regions the other way round: each one large enough to be a figure rather
than a rule or a logo becomes a figure, with the caption block nearest beneath it, and is
rendered from the original when the reader first asks for it.

Nothing is stored in the database and nothing is extracted at ingest. The original is
content-addressed in the store, so figures are a pure function of it; the rendered PNGs
and the index are a cache under `~/.library-agent/figures/<hash>/`, rebuilt if missing
and removed with the document. That also settles cartridges without a new field: the
`full` level ships the original, so the receiver's reader finds the same figures;
`readings` and `catalogue` ship no original and therefore no figures, which is what
those levels promise."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from library_agent.config import settings
from library_agent.ingest.extract import _figure_regions, _page_blocks

log = logging.getLogger(__name__)

INDEX_VERSION = 1
MIN_AREA_FRACTION = 0.03  # of the page: smaller is a rule, a logo, an inline glyph
MAX_AREA_FRACTION = 0.85  # larger is a scanned page or a full-page background
CAPTION_REACH = 72.0  # points below the region a caption may sit
ZOOM = 2.0


@dataclass
class Figure:
    n: int
    page: int  # 1-based, as pages are cited
    bbox: list[float]
    caption: str | None
    width: int
    height: int


def figures_dir(content_hash: str) -> Path:
    return settings().storage_dir.parent / "figures" / content_hash


def _caption_for(region: pymupdf.Rect, blocks: list):
    below = [
        b
        for b in blocks
        if b.kind == "caption"
        and b.rect.y0 >= region.y1 - 4
        and b.rect.y0 - region.y1 < CAPTION_REACH
        and b.rect.x1 > region.x0
        and b.rect.x0 < region.x1
    ]
    if not below:
        return None
    return min(below, key=lambda b: b.rect.y0)


def find_figures(pdf: Path) -> list[Figure]:
    out: list[Figure] = []
    with pymupdf.open(pdf) as doc:
        for page in doc:
            area = page.rect.get_area()
            if not area:
                continue
            try:
                regions = _figure_regions(page)
                blocks = _page_blocks(page)
            except Exception:
                log.debug(
                    "figure scan failed on page %s of %s", page.number, pdf.name, exc_info=True
                )
                continue
            for r in regions:
                r = r & page.rect
                frac = r.get_area() / area
                if frac < MIN_AREA_FRACTION or frac > MAX_AREA_FRACTION:
                    continue
                if r.width < 60 or r.height < 40:
                    continue
                # A little air around the clip, but stop above the caption: it is shown
                # as text beneath the image, where it can be selected and searched.
                clip = pymupdf.Rect(r.x0 - 6, r.y0 - 6, r.x1 + 6, r.y1 + 6) & page.rect
                cap = _caption_for(r, blocks)
                caption = " ".join(cap.text.split())[:400] if cap else None
                if cap and cap.rect.y0 - 2 > r.y1:
                    clip.y1 = min(clip.y1, cap.rect.y0 - 2)
                out.append(
                    Figure(
                        n=len(out) + 1,
                        page=page.number + 1,
                        bbox=[round(v, 1) for v in (clip.x0, clip.y0, clip.x1, clip.y1)],
                        caption=caption,
                        width=int(clip.width * ZOOM),
                        height=int(clip.height * ZOOM),
                    )
                )
    return out


def index(content_hash: str, pdf: Path) -> list[Figure]:
    """The figure list for a document, computed once and cached beside its renders."""
    d = figures_dir(content_hash)
    idx = d / "index.json"
    if idx.exists():
        try:
            data = json.loads(idx.read_text())
            if data.get("version") == INDEX_VERSION:
                return [Figure(**f) for f in data["figures"]]
        except Exception:
            log.debug("bad figure index for %s", content_hash, exc_info=True)
    if pdf.suffix.lower() != ".pdf" or not pdf.exists():
        return []
    figs = find_figures(pdf)
    d.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps({"version": INDEX_VERSION, "figures": [asdict(f) for f in figs]}))
    return figs


def render(content_hash: str, pdf: Path, n: int) -> Path | None:
    """The PNG for figure `n`, rendered from the original on first request."""
    out = figures_dir(content_hash) / f"{n}.png"
    if out.exists():
        return out
    figs = index(content_hash, pdf)
    fig = next((f for f in figs if f.n == n), None)
    if not fig:
        return None
    with pymupdf.open(pdf) as doc:
        page = doc[fig.page - 1]
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(ZOOM, ZOOM), clip=pymupdf.Rect(*fig.bbox), alpha=False
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        pix.save(out)
    return out


def forget(content_hash: str) -> None:
    """Drop the cache; called when the document goes."""
    shutil.rmtree(figures_dir(content_hash), ignore_errors=True)
