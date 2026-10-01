"""Markdown as the exporters write it, made fit to read and search.

Google Docs' "Download as Markdown" and Pandoc both write headings like
`# **Overview** {#overview}` -- the anchor and the bold are presentation, not the heading --
and Google embeds every image as base64 in a reference at the end of the file
(`[image1]: <data:image/png;base64,…>`). Read as-is, a 26-screenshot how-to guide became
1,315 passages, 1,306 of them base64, and took "**Overview** {#overview}" as its title.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass

# `[image1]: <data:image/png;base64,....>` -- a reference definition holding the image.
_DATA_REF = re.compile(r"^[ \t]*\[([^\]]+)\]:[ \t]*<?(data:[^\s>]*)>?[ \t]*$", re.MULTILINE)
# `![alt](data:image/png;base64,....)` -- an inline embedded image.
_DATA_INLINE = re.compile(r"!\[([^\]]*)\]\(\s*<?(data:[^)\s>]*)>?\s*\)")
_REF_USE = re.compile(r"!\[([^\]]*)\]\[([^\]]+)\]")
_DATA_URI = re.compile(r"^data:(image/[a-z0-9.+-]+)?(;[^,]*)?,(.*)$", re.IGNORECASE | re.DOTALL)


@dataclass
class Embedded:
    n: int  # figure number, by order of first use in the text
    alt: str
    mime: str
    data: bytes


def _decode(uri: str) -> tuple[str, bytes] | None:
    m = _DATA_URI.match(uri.strip())
    if not m or "base64" not in (m.group(2) or ""):
        return None
    try:
        return (m.group(1) or "image/png").lower(), base64.b64decode(m.group(3), validate=False)
    except (ValueError, TypeError):
        return None


def _uses(md: str) -> list[tuple[str, str]]:
    """Each embedded image where the text uses it, in order: (alt, data URI). A
    reference used twice is one image."""
    refs = {m.group(1): m.group(2) for m in _DATA_REF.finditer(md)}
    found: list[tuple[int, str, str]] = []
    for m in _DATA_INLINE.finditer(md):
        found.append((m.start(), m.group(1), m.group(2)))
    for m in _REF_USE.finditer(md):
        if m.group(2) in refs:
            found.append((m.start(), m.group(1), refs[m.group(2)]))
    out, seen = [], set()
    for _, alt, uri in sorted(found):
        if uri not in seen:
            seen.add(uri)
            out.append((alt, uri))
    return out


def embedded_images(md: str) -> list[Embedded]:
    """The images a Markdown file carries inside itself, numbered as the text uses them.
    These are its figures: kept, shown in the reader where they sat, and read by the
    vision model -- a how-to guide's screenshots are often the point."""
    out = []
    for n, (alt, uri) in enumerate(_uses(md), 1):
        if got := _decode(uri):
            out.append(Embedded(n=n, alt=" ".join(alt.split()), mime=got[0], data=got[1]))
    return out


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$", re.MULTILINE)
_ATTRS = re.compile(r"\s*\{[#.][^}]*\}\s*$")  # Pandoc / Google heading attributes
_EMPHASIS = re.compile(r"^(\*\*|__|\*|_)(.+)\1$")

# A first heading that names a part of a document, not the document.
GENERIC_TITLES = {
    "overview",
    "introduction",
    "intro",
    "contents",
    "table of contents",
    "summary",
    "background",
    "purpose",
    "scope",
    "abstract",
    "preface",
    "about",
    "readme",
}


def figure_marker(n: int, alt: str = "") -> str:
    """Where figure n sat in the text; the reader shows the image there."""
    alt = " ".join(alt.split())
    return f"*(Figure {n}{': ' + alt if alt else ''})*"


FIGURE_MARKER = re.compile(r"\*\(Figure (\d+)(?::[^)]*)?\)\*")


def clean_heading(text: str) -> str:
    """`**Overview** {#overview}` -> `Overview`."""
    t = _ATTRS.sub("", text).strip()
    while (m := _EMPHASIS.match(t)) and m.group(2).strip():
        t = m.group(2).strip()
    return t


def tidy(md: str) -> str:
    """Embedded images out of the text -- each replaced by a marker naming its figure
    (see embedded_images) -- heading attributes and wrapping emphasis off, empty headings
    gone. Everything else is left as written."""
    numbers = {uri: n for n, (_alt, uri) in enumerate(_uses(md), 1)}
    refs = {m.group(1): m.group(2) for m in _DATA_REF.finditer(md)}
    md = _DATA_REF.sub("", md)
    md = _DATA_INLINE.sub(lambda m: figure_marker(numbers.get(m.group(2), 0), m.group(1)), md)
    if refs:
        # `![][image1]` / `![alt][image1]` whose definition held the data.
        md = _REF_USE.sub(
            lambda m: (
                figure_marker(numbers[refs[m.group(2)]], m.group(1))
                if m.group(2) in refs
                else m.group(0)
            ),
            md,
        )

    def heading(m: re.Match) -> str:
        text = clean_heading(m.group(2))
        return f"{m.group(1)} {text}" if text else ""

    md = _HEADING.sub(heading, md)
    return re.sub(r"\n{3,}", "\n\n", md)


def is_generic_title(title: str | None) -> bool:
    return bool(title) and title.strip().strip(":").lower() in GENERIC_TITLES
