"""Text extraction. The spec gave this one line; in practice it is where most ingestion
pain lives, so the messy cases are handled explicitly: two-column papers, running
headers/footers, hyphenated line breaks, and PDFs with no usable text layer."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

log = logging.getLogger(__name__)

from library_agent.config import settings

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Ligatures and typographic junk that survive PDF extraction and poison lexical search.
_TRANSLATE = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
        "\u200b": "",
    }
)


@dataclass
class TocEntry:
    level: int
    title: str
    page: int  # 1-based
    # Exact character offset when the format gives it to us (markdown). PDF bookmarks
    # only carry a page number, so those stay None and get located by text search.
    char_start: int | None = None


@dataclass
class Extracted:
    text: str
    # (char_offset, page_number) sorted by offset; lets any char span resolve to a page.
    page_offsets: list[tuple[int, int]] = field(default_factory=list)
    toc: list[TocEntry] = field(default_factory=list)
    title: str | None = None
    page_count: int = 0
    needs_ocr: bool = False
    two_column_pages: int = 0

    def page_for_offset(self, offset: int) -> int | None:
        """Page containing a character offset. Used for citation attribution."""
        page = None
        for start, no in self.page_offsets:
            if start > offset:
                break
            page = no
        return page


def content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text: str) -> str:
    # Postgres rejects NUL in text columns outright, and other C0 controls are noise that
    # would otherwise reach the tsvector. Keep only newline and tab.
    text = _CONTROL_CHARS.sub("", text)
    text = unicodedata.normalize("NFKC", text.translate(_TRANSLATE))
    # Rejoin words split across a line break: "contex-\ntual" -> "contextual".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # A single newline inside a paragraph is a soft wrap; a blank line is a real break.
    text = re.sub(r"(?<![\n.!?:])\n(?![\n\s])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_two_column(blocks: list, page_width: float) -> bool:
    """Blocks sitting entirely on one side of the midline, on both sides, means columns.
    Checked per page because front matter is often single-column in a two-column paper."""
    mid = page_width / 2
    left = sum(1 for b in blocks if b[2] < mid * 1.05)
    right = sum(1 for b in blocks if b[0] > mid * 0.95)
    return left >= 2 and right >= 2 and (left + right) >= len(blocks) * 0.7


@dataclass
class _Block:
    rect: pymupdf.Rect
    text: str
    size: float
    kind: str  # body | caption | footnote


def _figure_regions(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Where the figures are: vector drawings and raster images. Text inside these
    regions is axis labels, node names and legend entries, not prose."""
    rects: list[pymupdf.Rect] = []
    try:
        rects += [pymupdf.Rect(d["rect"]) for d in page.get_drawings() if d.get("rect")]
    except Exception:
        log.debug("get_drawings failed on page %s", page.number, exc_info=True)
    try:
        for img in page.get_images(full=True):
            rects += page.get_image_rects(img[0])
    except Exception:
        log.debug("get_images failed on page %s", page.number, exc_info=True)
    # Merge into a few coarse regions so a label between two strokes still counts.
    merged: list[pymupdf.Rect] = []
    for r in sorted(rects, key=lambda r: (r.y0, r.x0)):
        if r.is_empty or r.get_area() < 4:
            continue
        for m in merged:
            if (m & r).get_area() > 0 or (
                abs(m.y0 - r.y0) < 24 and (m | r).width < page.rect.width
            ):
                m.include_rect(r)
                break
        else:
            merged.append(pymupdf.Rect(r))
    return merged


def _page_blocks(page: pymupdf.Page) -> list[_Block]:
    """Classify each text block using what the PDF actually knows: font sizes and
    positions. This is what separates prose from the figure labels, footnotes and
    superscript markers that plain text extraction fuses into it."""
    d = page.get_text("dict")
    sizes: dict[float, int] = {}
    for b in d["blocks"]:
        for line in b.get("lines", []):
            for s in line.get("spans", []):
                sizes[round(s["size"], 1)] = sizes.get(round(s["size"], 1), 0) + len(s["text"])
    if not sizes:
        return []
    body = max(sizes, key=sizes.get)
    figures = _figure_regions(page)
    h = page.rect.height

    out: list[_Block] = []
    for b in d["blocks"]:
        if b["type"] != 0:
            continue
        rect = pymupdf.Rect(b["bbox"])
        spans = [s for line in b["lines"] for s in line["spans"]]
        if not spans:
            continue
        size = max(s["size"] for s in spans)

        in_figure = any(
            rect.intersects(f) and (rect & f).get_area() > 0.5 * rect.get_area() for f in figures
        )
        # Figure text: small type inside a drawing, or a short label at any size inside one.
        raw_words = " ".join(s["text"] for s in spans).split()
        if in_figure and (size < body * 0.92 or len(raw_words) <= 4):
            continue

        is_footnote = size <= body * 0.88 and rect.y0 > h * 0.72 and not in_figure

        # Rebuild the text span by span so superscripts can be handled by their size.
        parts: list[str] = []
        for line in b["lines"]:
            line_parts: list[str] = []
            for i, s in enumerate(line["spans"]):
                txt = s["text"]
                tiny = s["size"] < size * 0.78 and txt.strip().isdigit() and len(txt.strip()) <= 2
                if tiny:
                    if is_footnote and i == 0:
                        line_parts.append(f"[{txt.strip()}] ")  # the footnote's own number
                    # In body prose a tiny digit is a reference marker: drop it.
                    continue
                line_parts.append(txt)
            parts.append("".join(line_parts))
        text = "\n".join(parts).strip()
        if not text:
            continue

        # Tables, axis ticks, equation fragments: almost no real words. Keep captions.
        words = text.split()
        prose = sum(1 for w in words if w.isalpha() and len(w) >= 3) / max(1, len(words))
        if (
            prose < 0.2
            and len(words) <= 14
            and not text[:7].lower().startswith(("figure", "table"))
        ):
            continue

        kind = "footnote" if is_footnote else "body"
        if text[:6].lower() in ("figure", "table ") or text.lower().startswith("table "):
            kind = "caption"
        out.append(_Block(rect=rect, text=text, size=size, kind=kind))
    return out


def _ordered_blocks(page: pymupdf.Page) -> list[str]:
    """Reading order. Naive top-to-bottom interleaves the columns of a two-column paper
    into nonsense, so detect columns and read each one fully before the next. Footnotes
    go after the page's prose, as notes, rather than wherever the layout put them."""
    blocks = _page_blocks(page)
    if not blocks:
        return []
    width = page.rect.width
    prose = [b for b in blocks if b.kind != "footnote"]
    notes = [b for b in blocks if b.kind == "footnote"]
    tuples = [(b.rect.x0, b.rect.y0, b.rect.x1, b.rect.y1, b.text, 0, 0) for b in prose]
    if _is_two_column(tuples, width):
        mid = width / 2
        left = sorted((b for b in prose if b.rect.x0 < mid), key=lambda b: b.rect.y0)
        right = sorted((b for b in prose if b.rect.x0 >= mid), key=lambda b: b.rect.y0)
        ordered = left + right
    else:
        ordered = sorted(prose, key=lambda b: (round(b.rect.y0, 1), b.rect.x0))
    result = [b.text for b in ordered]
    if notes:
        result.append(
            "\n".join(b.text for b in sorted(notes, key=lambda b: (b.rect.x0, b.rect.y0)))
        )
    return result


def _strip_running_heads(pages: list[list[str]], threshold: float = 0.5) -> list[list[str]]:
    """Page headers and footers repeat across most pages and add nothing but noise to
    every chunk. Detect by frequency, ignoring page numbers so they still match."""
    if len(pages) < 4:
        return pages

    def key(s: str) -> str:
        return re.sub(r"\d+", "#", s.strip().lower())[:80]

    first = Counter(key(p[0]) for p in pages if p)
    last = Counter(key(p[-1]) for p in pages if p)
    cutoff = len(pages) * threshold
    drop_first = {k for k, n in first.items() if n >= cutoff and len(k) < 80}
    drop_last = {k for k, n in last.items() if n >= cutoff and len(k) < 80}

    out = []
    for blocks in pages:
        b = list(blocks)
        if b and key(b[0]) in drop_first:
            b = b[1:]
        if b and key(b[-1]) in drop_last:
            b = b[:-1]
        out.append(b)
    return out


# Unreplaced LaTeX/Word template defaults that show up in real PDF metadata.
_PLACEHOLDER_TITLE = re.compile(
    r"^(untitled|no title|paper title|manuscript|document\d*|doc\d*|microsoft word|"
    r".*\bpaper title\b.*|.*\btemplate\b.*|.*\byour title\b.*)$",
    re.IGNORECASE,
)


def _usable_metadata_title(title: str | None) -> str | None:
    """Reject template junk like 'Transaction / Regular Paper Title' so the layout-based
    extractor gets a chance at the real title."""
    if not title:
        return None
    cleaned = normalize(title).strip()
    if len(cleaned) < 6 or _PLACEHOLDER_TITLE.match(cleaned):
        return None
    return cleaned[:300]


def _title_from_layout(page: pymupdf.Page) -> str | None:
    """Largest-font run on the first page. More reliable than PDF metadata, which is
    empty for most arXiv papers."""
    best_size, best_text = 0.0, ""
    try:
        data = page.get_text("dict")
    except Exception:  # noqa: BLE001 - a broken page must not fail the whole document
        return None
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            size = max((s.get("size", 0) for s in line.get("spans", [])), default=0)
            text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if not text or len(text) < 6:
                continue
            # Skip arXiv's rotated stamp down the left margin.
            if text.lower().startswith("arxiv:"):
                continue
            if size > best_size + 0.5:
                best_size, best_text = size, text
            elif abs(size - best_size) <= 0.5 and best_text and len(best_text) < 200:
                best_text += " " + text  # title wrapped onto a second line
    # Layout spans are joined with a space, so a title hyphenated across two lines
    # arrives as "LAN- GUAGE". normalize() only repairs the "-\n" form.
    cleaned = normalize(re.sub(r"(\w)-\s+(\w)", r"\1\2", best_text))
    return cleaned[:300] or None


def extract_pdf(path: Path) -> Extracted:
    doc = pymupdf.open(path)
    try:
        raw_pages = [_ordered_blocks(p) for p in doc]
        two_col = sum(
            1
            for p in doc
            if _is_two_column(
                [b for b in p.get_text("blocks") if b[6] == 0 and b[4].strip()], p.rect.width
            )
        )  # a cheap count for reporting; ordering itself uses the classified blocks
        pages = _strip_running_heads(raw_pages)

        chars = sum(len(" ".join(p)) for p in pages)
        needs_ocr = doc.page_count > 0 and (
            chars / doc.page_count < settings().ocr_char_per_page_threshold
        )

        parts: list[str] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for i, blocks in enumerate(pages, start=1):
            offsets.append((cursor, i))
            body = normalize("\n\n".join(blocks))
            parts.append(body)
            cursor += len(body) + 2

        toc = [
            TocEntry(level=max(1, lvl), title=normalize(t)[:300], page=pg)
            for lvl, t, pg in doc.get_toc(simple=True)
            if t and t.strip()
        ]
        title = _usable_metadata_title((doc.metadata or {}).get("title"))
        if not title and doc.page_count:
            title = _title_from_layout(doc[0])
        if not title:
            title = path.stem

        return Extracted(
            text="\n\n".join(parts),
            page_offsets=offsets,
            toc=toc,
            title=title,
            page_count=doc.page_count,
            needs_ocr=needs_ocr,
            two_column_pages=two_col,
        )
    finally:
        doc.close()


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.MULTILINE)


def extract_text_file(path: Path) -> Extracted:
    """Markdown and plaintext. ATX headings become the TOC; for plaintext there is none
    and the section builder falls back to windowing."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Strip fenced code blocks from heading detection but keep them in the body.
    text = normalize(raw) if path.suffix.lower() not in {".md", ".markdown"} else raw

    toc: list[TocEntry] = []
    if path.suffix.lower() in {".md", ".markdown"}:
        in_fence = False
        offset = 0
        for line in text.splitlines(keepends=True):
            stripped = line.rstrip("\n")
            start = offset
            offset += len(line)
            line = stripped
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if m := _MD_HEADING.match(line):
                toc.append(
                    TocEntry(
                        level=len(m.group(1)),
                        title=m.group(2).strip(),
                        page=1,
                        char_start=start + m.start(),
                    )
                )

    title = toc[0].title if toc and toc[0].level == 1 else path.stem
    return Extracted(text=text, page_offsets=[(0, 1)], toc=toc, title=title, page_count=1)


SUPPORTED = {".pdf", ".md", ".markdown", ".txt", ".text", ".rst"}


def extract(path: Path) -> Extracted:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in SUPPORTED:
        return extract_text_file(path)
    raise ValueError(f"unsupported file type: {suffix} (supported: {sorted(SUPPORTED)})")
