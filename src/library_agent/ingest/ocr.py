"""OCR for scanned PDFs, by the vision model.

A digital PDF carries its text; a scanned one carries pictures of pages, and extraction
finds almost nothing. Rather than refuse it, the library renders each page and asks the
vision model to transcribe it — a VLM reads a messy scan far better than a classical OCR
engine, and it is already the model the library runs for figures. The result is assembled
into the same `Extracted` shape a digital PDF produces, page offsets and all, so the rest
of the pipeline — sections, chunks, embeddings, citation by page — is unchanged.

It is slow: a page is a model call, so a book is many minutes. That is why it runs as a
background job, not inline with the upload."""

from __future__ import annotations

import logging

import pymupdf

from library_agent.config import settings
from library_agent.ingest.extract import Extracted, normalize

log = logging.getLogger(__name__)

OCR_PROMPT = (
    "Transcribe every word of text on this page, exactly as written, in reading order. "
    "Keep paragraph breaks, headings, lists, tables and code as they appear. Do not "
    "summarise, describe the layout, translate, correct, or add any commentary. If the "
    "page has no readable text, output nothing at all. Output only the transcribed text."
)


async def ocr_pdf(
    path,
    client,
    model: str,
    *,
    gate=None,
    progress=None,
    max_pages: int | None = None,
    dpi: int | None = None,
) -> Extracted:
    """Transcribe a PDF page by page with the vision model. Returns an `Extracted` with
    page offsets, so citations still resolve to a page."""
    cfg = settings()
    dpi = dpi or cfg.ocr_dpi
    doc = pymupdf.open(path)
    try:
        n = doc.page_count
        if max_pages:
            n = min(n, max_pages)
        parts: list[str] = []
        offsets: list[tuple[int, int]] = []
        pos = 0
        for i in range(n):
            if gate:
                await gate()
            png = doc[i].get_pixmap(dpi=dpi).tobytes("png")
            try:
                text = await client.describe_image(
                    model, OCR_PROMPT, png, num_predict=2048, temperature=0.0, timeout=240
                )
            except Exception:
                log.warning("OCR failed on page %d of %s", i + 1, path, exc_info=True)
                text = ""
            text = normalize(text or "").strip()
            offsets.append((pos, i + 1))
            parts.append(text)
            pos += len(text) + 2  # the "\n\n" the pages are joined with
            if progress:
                await progress(i + 1, n)
        title = (doc.metadata or {}).get("title") or None
        return Extracted(
            text="\n\n".join(parts),
            page_offsets=offsets,
            title=(title or "").strip() or None,
            page_count=doc.page_count,
            needs_ocr=False,
        )
    finally:
        doc.close()
