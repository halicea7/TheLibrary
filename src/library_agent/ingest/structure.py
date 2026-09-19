"""Section tree construction.

The spec treated a usable TOC as the common case and windowing as the exception. In
practice it is the other way round for scanned books and plain notes, so both paths are
first-class and the fallback is paragraph-aware rather than a blind fixed split."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from library_agent.ingest.extract import Extracted


@dataclass
class Section:
    order_index: int
    title: str | None
    level: int
    char_start: int
    char_end: int
    path: str = ""  # "Methods › Training setup", used in the chunk context prefix
    parent_index: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    children: list[int] = field(default_factory=list)


def _locate(text: str, title: str, hint: int, window: int = 12000) -> int | None:
    """Find a TOC title in the body near its recorded page.

    Anchored to line starts (with optional section numbering) because titles routinely
    occur in running prose first -- "Model Architecture" appears in the Transformer
    paper's introduction 2000 chars before the actual section heading."""
    if not title:
        return None
    lo = max(0, hint - 400)
    hi = min(len(text), hint + window)
    haystack = text[lo:hi]
    # TOC text may already carry its number ("3 Model Architecture") or not.
    bare = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title).strip()
    words = bare.split()
    if not words:
        return None
    body = r"\s+".join(re.escape(w) for w in words)
    numbering = r"(?:\d+(?:\.\d+)*\.?\s+)?"

    for pattern in (
        re.compile(rf"^[ \t]*{numbering}{body}\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(rf"^[ \t]*{numbering}{body}", re.IGNORECASE | re.MULTILINE),
    ):
        if m := pattern.search(haystack):
            return lo + m.start()

    # Distinctive prefix, still anchored to a line start.
    if len(words) > 3:
        short = r"\s+".join(re.escape(w) for w in words[:4])
        pat = re.compile(rf"^[ \t]*{numbering}{short}", re.IGNORECASE | re.MULTILINE)
        if m := pat.search(haystack):
            return lo + m.start()
    return None


def _assign_paths(sections: list[Section]) -> None:
    """Breadcrumb path plus parent links, from the level sequence."""
    stack: list[tuple[int, int]] = []  # (level, index)
    for s in sections:
        while stack and stack[-1][0] >= s.level:
            stack.pop()
        s.parent_index = stack[-1][1] if stack else None
        crumbs = [sections[i].title or "" for _, i in stack] + [s.title or ""]
        s.path = " › ".join(c for c in crumbs if c)
        if s.parent_index is not None:
            sections[s.parent_index].children.append(s.order_index)
        stack.append((s.level, s.order_index))


def _prune_short(sections: list[Section], min_chars: int = 120) -> list[Section]:
    """Drop trivially short sections, but keep any that head a deeper subsection.

    A parent heading immediately followed by its first subheading spans almost no text of
    its own ("3 Deep Residual Learning" then "3.1 ..."). Dropping it reparents 3.1 under
    the previous unrelated section, and that wrong breadcrumb gets embedded into every
    chunk's context prefix."""
    keep: list[Section] = []
    for i, s in enumerate(sections):
        is_parent = i + 1 < len(sections) and sections[i + 1].level > s.level
        if is_parent or (s.char_end - s.char_start) >= min_chars:
            keep.append(s)
    return keep


_PARA = re.compile(r"\n\s*\n")

# "3 Deep Residual Learning", "3.1. Residual Learning"
_NUMBERED_HEADING = re.compile(
    r"^[ \t]*(\d+(?:\.\d+){0,3})\.?[ \t]+([A-Z][^\n]{2,70})[ \t]*$", re.MULTILINE
)
# "ABSTRACT", "RELATED WORK"
_CAPS_HEADING = re.compile(r"^[ \t]*([A-Z][A-Z0-9 \t&/-]{3,50})[ \t]*$", re.MULTILINE)


def detect_headings(text: str) -> list[tuple[int, int, str]]:
    """(char_start, level, title) scanned from the body.

    Most PDFs have no bookmarks -- the ResNet paper has zero -- but nearly all papers
    still render numbered headings, which is far better structure than blind windowing."""
    found: list[tuple[int, int, str]] = []
    for m in _NUMBERED_HEADING.finditer(text):
        number, title = m.group(1), m.group(2).strip()
        if title.endswith((".", ",", ";")) or len(title.split()) > 12:
            continue  # a sentence or a numbered list item, not a heading
        found.append((m.start(), number.count(".") + 1, f"{number} {title}"))
    for m in _CAPS_HEADING.finditer(text):
        title = m.group(1).strip()
        if len(title.split()) > 6 or not any(c.isalpha() for c in title):
            continue
        # "BERT", "E1 E2", "BERT BERT" are figure labels and table headers, not headings.
        # Real all-caps headings ("ABSTRACT", "RELATED WORK") carry a word of 5+ letters.
        if not re.search(r"[A-Za-z]{5,}", title):
            continue
        found.append((m.start(), 1, title.title()))

    found.sort(key=lambda x: x[0])
    deduped: list[tuple[int, int, str]] = []
    for item in found:
        # Small window: only collapse the same heading matched by both patterns.
        # A heading immediately followed by its first subheading is legitimate.
        if not deduped or item[0] - deduped[-1][0] > 25:
            deduped.append(item)
    return deduped


def _window_sections(text: str, target_chars: int = 6000) -> list[Section]:
    """Fallback for documents with no usable structure. Splits on paragraph boundaries
    rather than mid-sentence, which keeps chunks coherent downstream."""
    if not text.strip():
        return []
    bounds = [0, *[m.end() for m in _PARA.finditer(text)], len(text)]
    sections: list[Section] = []
    start = 0
    idx = 0
    for b in bounds[1:]:
        if b - start >= target_chars or b == len(text):
            if b - start < 200 and sections:  # absorb a trailing runt
                sections[-1].char_end = b
            else:
                sections.append(
                    Section(order_index=idx, title=None, level=1, char_start=start, char_end=b)
                )
                idx += 1
            start = b
    return sections


def build_sections(ex: Extracted) -> list[Section]:
    """TOC-derived where possible, paragraph-windowed otherwise."""
    text = ex.text
    entries = ex.toc

    if entries:
        page_offset = {no: off for off, no in ex.page_offsets}
        located: list[tuple[int, int, str]] = []  # (char_start, level, title)
        for e in entries:
            if e.char_start is not None:
                located.append((e.char_start, e.level, e.title))
                continue
            hint = page_offset.get(e.page, 0)
            pos = _locate(text, e.title, hint)
            located.append((pos if pos is not None else hint, e.level, e.title))

        # TOC order must be monotonic in the body; drop entries that resolved backwards
        # (usually a title string that also appears in the reference list).
        cleaned: list[tuple[int, int, str]] = []
        for item in sorted(located, key=lambda x: x[0]):
            if not cleaned or item[0] > cleaned[-1][0]:
                cleaned.append(item)

        if cleaned:
            sections: list[Section] = []
            # Front matter before the first heading (abstract, authors) is worth keeping.
            if cleaned[0][0] > 400:
                sections.append(
                    Section(
                        order_index=0,
                        title="Front matter",
                        level=1,
                        char_start=0,
                        char_end=cleaned[0][0],
                    )
                )
            for i, (start, level, title) in enumerate(cleaned):
                end = cleaned[i + 1][0] if i + 1 < len(cleaned) else len(text)
                sections.append(
                    Section(
                        order_index=len(sections),
                        title=title,
                        level=level,
                        char_start=start,
                        char_end=end,
                    )
                )
            sections = _prune_short(sections)
            for i, s in enumerate(sections):
                s.order_index = i
                s.children = []
            if sections:
                _assign_paths(sections)
                for s in sections:
                    s.page_start = ex.page_for_offset(s.char_start)
                    s.page_end = ex.page_for_offset(max(s.char_start, s.char_end - 1))
                return sections

    detected = detect_headings(text)
    if len(detected) >= 3:
        sections = []
        if detected[0][0] > 400:
            sections.append(
                Section(
                    order_index=0,
                    title="Front matter",
                    level=1,
                    char_start=0,
                    char_end=detected[0][0],
                )
            )
        for i, (start, level, title) in enumerate(detected):
            end = detected[i + 1][0] if i + 1 < len(detected) else len(text)
            sections.append(
                Section(
                    order_index=len(sections),
                    title=title,
                    level=level,
                    char_start=start,
                    char_end=end,
                )
            )
        sections = _prune_short(sections)
        for i, s in enumerate(sections):
            s.order_index, s.children = i, []
    else:
        sections = _window_sections(text)
    _assign_paths(sections)
    for s in sections:
        s.page_start = ex.page_for_offset(s.char_start)
        s.page_end = ex.page_for_offset(max(s.char_start, s.char_end - 1))
    return sections
