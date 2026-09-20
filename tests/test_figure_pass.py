"""Figures read by the vision model become passages of kind 'figure'."""

from pathlib import Path

import pytest
from conftest import BODY_A, make_document
from sqlalchemy import select

from library_agent.db.models import Chunk, Embedding, OwnerKind
from library_agent.reading import figures as fp


@pytest.fixture
def paper(tmp_path: Path) -> Path:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 90), "Results are shown below.", fontsize=11)
    s = page.new_shape()
    s.draw_rect(pymupdf.Rect(90, 140, 500, 420))
    s.finish(color=(0, 0, 0), width=1.2)
    s.commit()
    page.insert_text((90, 445), "Figure 1: Recall against depth.", fontsize=9)
    p = tmp_path / "paper.pdf"
    doc.save(p)
    return p


async def test_figures_become_passages_and_replace_on_rerun(db, paper, monkeypatch):
    from library_agent import config

    monkeypatch.setattr(config.settings(), "storage_dir", paper.parent / "documents")
    doc = await make_document(db, title="Recall Paper", body=BODY_A)
    doc.source_path = str(paper)
    await db.flush()

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "A line chart of recall rising with depth, flattening past twenty."}

    class FakeHttp:
        async def post(self, *a, **k):
            return FakeResp()

    class FakeClient:
        _client = FakeHttp()

        async def embed(self, texts, model=None):
            return [[0.01] * 1024 for _ in texts]

        async def aclose(self):
            pass

    async def fake_embed(texts, client=None):
        return [[0.01] * 1024 for _ in texts]

    monkeypatch.setattr(fp, "embed_texts", fake_embed)
    r = await fp.describe_figures(db, doc.id, client=FakeClient())
    assert r.figures == 1 and r.described == 1
    rows = list(
        (
            await db.execute(
                select(Chunk).where(Chunk.document_id == doc.id, Chunk.kind == "figure")
            )
        ).scalars()
    )
    assert len(rows) == 1
    assert rows[0].text.startswith("Figure 1, p.1: Recall against depth.")
    assert "line chart of recall" in rows[0].text
    assert rows[0].page_start == 1 and rows[0].order_index >= fp.FIGURE_ORDER_BASE
    assert rows[0].id == fp.figure_chunk_id(doc.content_hash, 1)
    vec = (
        await db.execute(
            select(Embedding).where(
                Embedding.owner_kind == OwnerKind.CHUNK, Embedding.owner_id == rows[0].id
            )
        )
    ).scalar_one_or_none()
    assert vec is not None

    # A second pass replaces rather than duplicates.
    await fp.describe_figures(db, doc.id, client=FakeClient())
    n = len(
        list(
            (
                await db.execute(
                    select(Chunk.id).where(Chunk.document_id == doc.id, Chunk.kind == "figure")
                )
            ).scalars()
        )
    )
    assert n == 1


async def test_non_pdf_has_nothing_to_describe(db):
    doc = await make_document(db, title="Notes", body=BODY_A)
    r = await fp.describe_figures(db, doc.id, client=object())
    assert r.figures == 0 and r.described == 0
