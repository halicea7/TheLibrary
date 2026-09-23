"""OCR assembles the vision model's per-page transcription into the same Extracted a
digital PDF produces — text joined, page offsets kept, so citations still resolve."""

from __future__ import annotations

import pymupdf
import pytest

from library_agent.ingest import ocr


class FakeVision:
    """Returns a fixed transcription per page, and records the prompt it was asked."""

    def __init__(self):
        self.calls = 0

    async def describe_image(self, model, prompt, png, **kw):
        self.calls += 1
        assert "Transcribe" in prompt and isinstance(png, bytes) and png[:4] == b"\x89PNG"
        return f"Text of page {self.calls}."


@pytest.fixture
def image_pdf(tmp_path):
    # two image-only pages (no text layer)
    src = pymupdf.open()
    doc = pymupdf.open()
    for _ in range(2):
        p = src.new_page(width=200, height=200)
        p.insert_text((20, 60), "rendered", fontsize=14)
    for i in range(2):
        pm = src[i].get_pixmap(dpi=72)
        page = doc.new_page(width=pm.width, height=pm.height)
        page.insert_image(page.rect, pixmap=pm)
    path = tmp_path / "scan.pdf"
    doc.save(path)
    return path


async def test_ocr_pdf_assembles_pages_with_offsets(image_pdf):
    c = FakeVision()
    ex = await ocr.ocr_pdf(image_pdf, c, "vision-model", dpi=72)
    assert c.calls == 2
    assert ex.text == "Text of page 1.\n\nText of page 2."
    assert ex.needs_ocr is False and ex.page_count == 2
    # page 2 starts after page 1's text plus the blank line, and resolves to page 2
    assert ex.page_offsets[0] == (0, 1)
    assert ex.page_offsets[1][1] == 2
    assert ex.page_for_offset(ex.page_offsets[1][0]) == 2


async def test_a_failed_page_does_not_sink_the_document(image_pdf):
    class Flaky:
        def __init__(self):
            self.n = 0

        async def describe_image(self, model, prompt, png, **kw):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("vision timed out")
            return "Second page text."

    ex = await ocr.ocr_pdf(image_pdf, Flaky(), "m", dpi=72)
    assert "Second page text." in ex.text  # page 2 survived page 1's failure
