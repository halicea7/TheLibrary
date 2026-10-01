"""HTML pages as Markdown, so a wiki export drops straight onto the shelf.

The motivating case is a Confluence space export: one HTML file per page, the page's
own body inside `#main-content`, and around it the export's chrome -- breadcrumbs, the
space name, a footer with the export date -- which is not the document and must not be
read as one. The body is converted to Markdown with headings, lists, code blocks, links
and tables kept, and the page title becomes the H1, so from here on the page is a
Markdown volume: the section tree comes from its headings, the reader renders it as it
was written, and nothing downstream knows it was ever HTML."""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

# Where the page body lives, most specific first. Confluence (both the old export and
# the new), MediaWiki, common static-site generators, then the HTML5 landmarks.
_BODY_SELECTORS = [
    "#main-content",
    "div.wiki-content",
    "#mw-content-text",
    "article",
    "main",
    '[role="main"]',
    "#content",
    "body",
]
# Chrome that sits inside the body on some exports.
_STRIP_SELECTORS = [
    "nav",
    "header",
    "footer",
    "script",
    "style",
    "noscript",
    "#breadcrumb-section",
    "#breadcrumbs",
    ".breadcrumbs",
    "#footer",
    ".footer",
    "#navigation",
    ".pageSection.group",  # Confluence: attachments / comments blocks
    ".page-metadata",
    ".toc-macro",
]
_TITLE_SPLIT = re.compile(r"\s+[:|–—-]\s+")


class _Converter(MarkdownConverter):
    """ATX headings, fenced code, and no escaping noise in prose."""

    def convert_pre(self, el, text, parent_tags):
        code = el.get_text()
        lang = ""
        # Confluence marks the language on the pre's data attribute; GitHub-style
        # exports put it on a class of the inner code element.
        if el.get("data-syntaxhighlighter-params"):
            m = re.search(r"brush:\s*([\w+-]+)", el["data-syntaxhighlighter-params"])
            lang = m.group(1) if m else ""
        elif (inner := el.find("code")) and inner.get("class"):
            for c in inner["class"]:
                if c.startswith(("language-", "lang-")):
                    lang = c.split("-", 1)[1]
        return f"\n\n```{lang}\n{code.rstrip()}\n```\n\n"


def _unsited(t: str) -> str:
    """ "Space Name : Page Title" (Confluence), "Page Title - Site" (most others): the
    page's own name is the longest part."""
    parts = _TITLE_SPLIT.split(t)
    return max(parts, key=len).strip() if len(parts) > 1 else t


def page_title(soup: BeautifulSoup) -> str | None:
    h1 = soup.select_one("#title-text, h1.pagetitle, h1")
    if h1 and h1.get_text(strip=True):
        return _unsited(" ".join(h1.get_text().split()))
    if soup.title and soup.title.get_text(strip=True):
        return _unsited(" ".join(soup.title.get_text().split()))
    return None


def to_markdown(html: str) -> tuple[str, str | None]:
    """(markdown, title). The title is the H1 of the result when there is one."""
    soup = BeautifulSoup(html, "html.parser")
    title = page_title(soup)
    body = None
    for sel in _BODY_SELECTORS:
        body = soup.select_one(sel)
        if body is not None:
            break
    if body is None:
        body = soup
    for sel in _STRIP_SELECTORS:
        for el in body.select(sel):
            el.decompose()
    # The page's own title heading would repeat the H1 we add.
    for h in body.select("#title-text, h1.pagetitle"):
        h.decompose()
    md = _Converter(
        heading_style="ATX", bullets="-", escape_asterisks=False, escape_underscores=False
    ).convert_soup(body)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if title:
        md = f"# {title}\n\n{md}"
    return md + "\n", title


def extract_html(path: Path) -> str:
    return to_markdown(path.read_text(encoding="utf-8", errors="replace"))[0]


def page_people(html: str) -> list[str]:
    """Who made a wiki page: Confluence's "Created by <author> on …, last updated by
    <editor> on …", the creator first. The block is stripped from the body (it isn't the
    document) but kept here, so "what has Hector written?" can be answered."""
    soup = BeautifulSoup(html, "html.parser")
    block = soup.select_one(".page-metadata")
    if block is None:
        return []
    names = [
        " ".join(el.get_text().split())
        for el in block.select(".author, .editor, [class*=author], [class*=editor]")
    ]
    if not names:  # an older export: "Created by Ann Lee, last modified by Bo Ng on …"
        names = re.findall(
            r"(?:created|modified|updated) by\s+([^,<]+?)(?:\s+on\b|,|$)",
            block.get_text(),
            re.IGNORECASE,
        )
    return list(dict.fromkeys(n.strip() for n in names if n and n.strip()))[:8]


def page_people_of(path: Path) -> list[str]:
    try:
        return page_people(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return []


async def backfill_authors(session_scope) -> dict:
    """Read who made each wiki page already on the shelf from its stored original, for
    volumes ingested before authors were kept. Volumes that already have authors, or
    whose original isn't an HTML page on this machine, are left alone."""
    from sqlalchemy import select

    from library_agent.db.models import Document

    found = missing = 0
    async with session_scope() as db:
        docs = list(
            (
                await db.execute(
                    select(Document).where(
                        Document.authors.is_(None), Document.source_path.ilike("%.htm%")
                    )
                )
            ).scalars()
        )
        for d in docs:
            people = page_people_of(Path(d.source_path))
            if people:
                d.authors = people
                found += 1
            else:
                missing += 1
    return {"volumes_given_authors": found, "html_without_a_byline": missing}


def main() -> None:
    import asyncio

    from library_agent.db.session import session_scope

    print(asyncio.run(backfill_authors(session_scope)))


if __name__ == "__main__":
    main()
