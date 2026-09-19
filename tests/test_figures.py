"""Figures are a pure function of the original: found by region, captioned by the
nearest caption block beneath, rendered on demand, forgotten with the document."""

from pathlib import Path

import pymupdf
import pytest

from library_agent.ingest import figures


@pytest.fixture
def paper(tmp_path: Path) -> Path:
    """A one-page paper: a paragraph, a drawn chart, its caption, more prose."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        (72, 90), "Retrieval quality is measured by mean reciprocal rank.", fontsize=11
    )
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(90, 140, 500, 420))
    for i in range(6):
        shape.draw_line((110 + i * 60, 400 - i * 35), (170 + i * 60, 380 - i * 30))
    shape.finish(color=(0, 0, 0), width=1.2)
    shape.commit()
    page.insert_text(
        (90, 445), "Figure 1: Recall against depth for three fusion weights.", fontsize=9
    )
    page.insert_text(
        (72, 520), "The curve flattens after depth twenty in every setting.", fontsize=11
    )
    # A thin rule: too small to be a figure.
    r = page.new_shape()
    r.draw_line((72, 780), (520, 780))
    r.finish(color=(0, 0, 0), width=0.5)
    r.commit()
    p = tmp_path / "paper.pdf"
    doc.save(p)
    return p


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch):
    from library_agent import config

    monkeypatch.setattr(config.settings(), "storage_dir", tmp_path / "documents")


def test_finds_the_chart_with_its_caption(paper: Path):
    figs = figures.find_figures(paper)
    assert len(figs) == 1, figs
    f = figs[0]
    assert f.page == 1 and f.n == 1
    assert f.caption and f.caption.startswith("Figure 1: Recall against depth")
    # The clip stops above the caption line rather than taking it in.
    assert f.bbox[3] < 445
    assert f.width > 700 and f.height > 500  # 2x render


def test_index_is_cached_and_render_writes_png(paper: Path):
    h = "a" * 64
    first = figures.index(h, paper)
    assert (figures.figures_dir(h) / "index.json").exists()
    paper.unlink()  # the cache answers even when the original is gone
    assert [f.n for f in figures.index(h, paper)] == [f.n for f in first]


def test_render_and_forget(paper: Path):
    h = "b" * 64
    png = figures.render(h, paper, 1)
    assert png and png.exists() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert figures.render(h, paper, 9) is None
    figures.forget(h)
    assert not figures.figures_dir(h).exists()


def test_not_a_pdf_has_no_figures(tmp_path: Path):
    md = tmp_path / "notes.md"
    md.write_text("# Notes\n\n![chart](chart.png)\n")
    assert figures.index("c" * 64, md) == []
