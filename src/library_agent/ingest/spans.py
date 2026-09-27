"""Do stored passages open on the text they cite?

A chunk records where it sits in its document's extraction (`char_start`, `char_end`,
`page_start`). Until 2026-09-27 that span was found by searching for the chunk's opening
characters, and one chunk in six came back at the wrong place or length. This module
checks every chunk against a fresh extraction of its original and, with `repair`, moves
the span to where the chunk actually is. Chunk ids do not change -- they are identifiers
now, whatever span they were first derived from -- so embeddings, readings and citations
that point at a chunk still find it.

    uv run python -m library_agent.ingest.spans            # report
    uv run python -m library_agent.ingest.spans --repair   # fix, and mark each chunk

CPU only: extraction runs locally, nothing is embedded or generated."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from library_agent.db.models import Chunk, Document
from library_agent.db.session import session_scope
from library_agent.ingest.chunk import _Aligner
from library_agent.ingest.extract import extract

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def _dense(s: str) -> str:
    return _WS.sub("", s)


@dataclass
class SpanReport:
    documents: int = 0
    chunks: int = 0
    exact: int = 0
    repaired: int = 0
    unresolved: int = 0
    skipped_documents: list[str] = field(default_factory=list)


async def check_document(db, doc: Document, *, repair: bool, report: SpanReport) -> None:
    path = Path(doc.source_path)
    if doc.readings_only or not path.exists():
        report.skipped_documents.append(doc.title)
        return
    try:
        ex = extract(path)
    except Exception as exc:  # noqa: BLE001 -- a parser change or a lost dependency: say so, move on
        log.warning("could not re-extract %s: %s", doc.title, exc)
        report.skipped_documents.append(doc.title)
        return
    chunks = (
        (
            await db.execute(
                select(Chunk)
                .where(Chunk.document_id == doc.id, Chunk.kind == "text")
                .order_by(Chunk.order_index)
            )
        )
        .scalars()
        .all()
    )
    report.documents += 1
    aligner = _Aligner(ex.text)
    cursor = 0
    for c in chunks:
        report.chunks += 1
        if _dense(ex.text[c.char_start : c.char_end]) == _dense(c.text):
            report.exact += 1
            cursor = c.char_start + 1
            if repair and c.span_exact is not True:
                c.span_exact = True
            continue
        # In order, from the last chunk placed; then anywhere, for a chunk out of order.
        span = aligner.find(c.text, cursor, len(ex.text)) or aligner.find(c.text, 0, len(ex.text))
        if span is None:
            report.unresolved += 1
            if repair:
                c.span_exact = False
            continue
        report.repaired += 1
        cursor = span[0] + 1
        if repair:
            c.char_start, c.char_end = span
            c.page_start = ex.page_for_offset(span[0])
            c.span_exact = True


async def check_all(*, repair: bool = False, limit: int | None = None) -> SpanReport:
    report = SpanReport()
    async with session_scope() as db:
        ids = (await db.execute(select(Document.id).order_by(Document.added_at))).scalars().all()
    for i, doc_id in enumerate(ids[:limit] if limit else ids):
        # One transaction per document, so a long run commits as it goes.
        async with session_scope() as db:
            doc = await db.get(Document, doc_id)
            if doc:
                await check_document(db, doc, repair=repair, report=report)
        if (i + 1) % 100 == 0:
            log.info("checked %d volumes", i + 1)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repair", action="store_true", help="move each span to where the chunk is")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = asyncio.run(check_all(repair=a.repair, limit=a.limit))
    verb = "repaired" if a.repair else "misplaced (fixable)"
    print(
        f"{r.documents} volumes, {r.chunks} passages: {r.exact} exact, {r.repaired} {verb}, "
        f"{r.unresolved} not found in the text"
        + (f"; {len(r.skipped_documents)} volumes skipped" if r.skipped_documents else "")
    )


if __name__ == "__main__":
    main()
