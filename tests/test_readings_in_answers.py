"""Ask is given the library's reading beside the passages; a named volume is not capped;
the owner's note reaches the prompt."""

from __future__ import annotations

import uuid

import pytest
from conftest import _vec, make_document

from library_agent.chat import answer
from library_agent.chat.citations import build_sources, render_context
from library_agent.llm import providers
from library_agent.retrieval import readings
from library_agent.retrieval.hybrid import SearchHit
from library_agent.retrieval.pipeline import diversify


async def test_a_question_names_its_volume(scratch_db):
    db = scratch_db
    book = await make_document(db, title="The 48 Laws of Power", body="power " * 80)
    await make_document(db, title="Kernel Notes", body="kernel " * 80)
    await make_document(db, title="Power", body="short " * 80)
    for q in (
        "How would you use The 48 Laws of Power here?",
        "what does 48 laws of power say about envy",
        "Summarise THE 48 LAWS OF POWER.",
    ):
        assert await readings.named_documents(db, q) == [book.id]
    # a title met in passing still names its volume; a word of it does not
    assert len(await readings.named_documents(db, "how do kernel notes work")) == 1
    assert await readings.named_documents(db, "how does the kernel schedule") == []
    assert await readings.named_documents(db, "who holds power in a team") == []


async def test_readings_come_from_the_named_volume_first(scratch_db, monkeypatch):
    db = scratch_db
    a = await make_document(db, title="Alpha Volume", body="alpha " * 80)
    b = await make_document(db, title="Beta Volume", body="beta " * 80)

    async def near_part_zero(text, client=None):
        return _vec(20)  # the reading of each volume's first section

    monkeypatch.setattr(readings, "embed_query", near_part_zero)
    hits = await readings.retrieve_readings(db, "q", limit=4)
    assert hits and all(h.kind == "reading" for h in hits)
    assert hits[0].text.startswith("Reading of part 0")
    assert {h.document_id for h in hits} == {a.id, b.id}

    fav = await readings.retrieve_readings(db, "q", limit=4, favour=[b.id])
    assert [h.document_id for h in fav[:2]] == [b.id, b.id]
    assert len({(h.document_id, h.section_path) for h in fav}) == len(fav)
    # a scope holds: readings from outside it never appear
    only_a = await readings.retrieve_readings(db, "q", limit=4, document_ids=[a.id], favour=[b.id])
    assert {h.document_id for h in only_a} == {a.id}


def _hit(doc, n):
    return SearchHit(
        chunk_id=uuid.uuid4(),
        document_id=doc,
        document_title="t",
        section_path="",
        page=1,
        text=f"p{n}",
        score=1.0 - n / 100,
        dense_rank=None,
        lexical_rank=None,
    )


def test_the_named_volume_is_not_capped():
    book, other = uuid.uuid4(), uuid.uuid4()
    hits = [_hit(book, i) for i in range(6)] + [_hit(other, 10 + i) for i in range(4)]
    capped = diversify(hits, per_document=2, limit=6)
    assert sum(h.document_id == book for h in capped[:4]) == 2
    free = diversify(hits, per_document=2, limit=6, exempt={book})
    assert [h.document_id for h in free] == [book] * 6


def test_a_reading_says_what_it_is():
    p = _hit(uuid.uuid4(), 1)
    r = _hit(uuid.uuid4(), 2)
    r.kind, r.section_path, r.text = "reading", "LAW 14 - POSE AS A FRIEND", "The law says..."
    srcs = build_sources([p, r])
    assert [s.kind for s in srcs] == ["passage", "reading"]
    ctx = render_context([p, r], srcs)
    assert "[2] the library's reading of t, p.1 — LAW 14" in ctx
    assert "[1] t, p.1" in ctx


@pytest.fixture
def pfile(tmp_path, monkeypatch):
    p = tmp_path / "providers.json"
    monkeypatch.setenv("LIBRARY_PROVIDERS_FILE", str(p))
    providers._cache = None
    return p


def test_about_you_reaches_the_prompt(pfile):
    msgs = answer.build_messages("q", [], [], [])
    assert "About the owner" not in msgs[0]["content"]
    cfg = providers.load()
    cfg.about = "I work in IT security; offensive technique is part of the job."
    providers.save(cfg)
    assert providers.load().about.startswith("I work in IT security")
    msgs = answer.build_messages("q", [], [], [])
    assert "offensive technique is part of the job" in msgs[0]["content"]


async def test_about_you_route(pfile):
    import httpx

    from library_agent.api.app import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        assert (await c.get("/api/settings/about")).json()["about"] == ""
        r = await c.put("/api/settings/about", json={"about": "  A network engineer.  "})
        assert r.json()["about"] == "A network engineer."
        assert (await c.put("/api/settings/about", json={"about": "x" * 2001})).status_code == 422
        # the rest of the file is untouched by it
        cfg = providers.load()
        assert cfg.about == "A network engineer." and cfg.models == {}
