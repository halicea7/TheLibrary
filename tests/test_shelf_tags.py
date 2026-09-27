"""Shelf placement is a home, not a subject: moving a volume changes its shelf tag and
nothing the reader said about it, and a twice-reshelved volume carries one shelf tag."""

from __future__ import annotations

import uuid

from conftest import make_document
from sqlalchemy import select

from library_agent.db.models import Artifact, ArtifactKind, Category, DocumentCategory
from library_agent.library import shelving, taxonomy


async def _tags(db, doc_id):
    rows = (
        await db.execute(
            select(Category.name, DocumentCategory.origin)
            .join(DocumentCategory, DocumentCategory.category_id == Category.id)
            .where(DocumentCategory.document_id == doc_id)
        )
    ).all()
    return sorted((n, o) for n, o in rows)


async def _shelves(db):
    top = await taxonomy.get_or_create(db, "Distributed Systems")
    a = await taxonomy.get_or_create(db, "Consensus")
    b = await taxonomy.get_or_create(db, "Replication")
    a.parent_id = b.parent_id = top.id
    other_top = await taxonomy.get_or_create(db, "Web Security")
    stray = await taxonomy.get_or_create(db, "Session Handling")
    stray.parent_id = other_top.id
    await db.flush()
    return top, a, b, other_top, stray


async def test_moving_a_volume_replaces_only_its_shelf_tag(scratch_db):
    db = scratch_db
    doc = await make_document(db, title="Raft", body="raft " * 40)
    await taxonomy.assign_document(db, doc.id, ["Consensus", "Leader Election"])
    _, a, b, _, _ = await _shelves(db)
    doc.shelf_id = a.id
    await shelving._finish_place(db, doc, a)  # its home is also a reader subject
    doc.shelf_id = b.id
    await shelving._finish_place(db, doc, b)
    doc.shelf_id = a.id
    await shelving._finish_place(db, doc, a)
    doc.shelf_id = b.id
    await shelving._finish_place(db, doc, b)  # twice reshelved
    assert await _tags(db, doc.id) == [
        ("Consensus", "reader"),  # the reader named it: it stays when the volume moves
        ("Leader Election", "reader"),
        ("Replication", "shelf"),  # the one shelf tag, the current home's
    ]


async def test_a_re_read_keeps_the_home_tag(scratch_db):
    db = scratch_db
    doc = await make_document(db, title="Paxos", body="paxos " * 40)
    _, _a, b, _, _ = await _shelves(db)
    await taxonomy.assign_document(db, doc.id, ["Replication"])
    doc.shelf_id = b.id
    await shelving._finish_place(db, doc, b)
    await taxonomy.assign_document(db, doc.id, ["Consensus"])  # re-read, new subjects
    assert await _tags(db, doc.id) == [("Consensus", "reader"), ("Replication", "shelf")]


async def test_a_sub_shelf_under_another_top_is_refused(scratch_db):
    db = scratch_db
    top, a, b, other_top, stray = await _shelves(db)
    tx = shelving.Taxonomy(
        tops={"Distributed Systems": ["Consensus", "Replication"]},
        ids={
            "Distributed Systems": top.id,
            "Consensus": a.id,
            "Replication": b.id,
            # the id map is shared across collections, so another's names are in it
            "Web Security": other_top.id,
            "Session Handling": stray.id,
        },
    )
    ok = await shelving._resolve_place(
        db, tx, {"top_shelf": "Distributed Systems", "sub_shelf": "Consensus"}
    )
    assert ok and ok.id == a.id
    # a sub-shelf of a top shelf not in view: another collection's
    assert (
        await shelving._resolve_place(
            db, tx, {"top_shelf": "Distributed Systems", "sub_shelf": "Session Handling"}
        )
        is None
    )
    # in view, but not under the top shelf the answer named
    tx.tops["Web Security"] = ["Session Handling"]
    assert (
        await shelving._resolve_place(
            db, tx, {"top_shelf": "Distributed Systems", "sub_shelf": "Session Handling"}
        )
        is None
    )


async def test_stale_shelf_tags_are_found_and_reader_subjects_kept(scratch_db):
    db = scratch_db
    doc = await make_document(db, title="Chubby", body="lock " * 40)
    _, a, b, _, _ = await _shelves(db)
    summary = (
        await db.execute(
            select(Artifact).where(
                Artifact.kind == ArtifactKind.DOCUMENT_SUMMARY, Artifact.target_id == doc.id
            )
        )
    ).scalar_one()
    summary.data = {"categories": ["Consensus"]}
    # Tags from before origins: the reader's, the current home, and an old home.
    old_home = await taxonomy.get_or_create(db, "Coordination Services")
    await db.execute(
        DocumentCategory.__table__.delete().where(DocumentCategory.document_id == doc.id)
    )
    for c in (a, b, old_home):
        db.add(DocumentCategory(document_id=doc.id, category_id=c.id))
    doc.shelf_id = b.id
    await db.flush()
    r = await shelving.classify_tags(db, repair=True)
    assert (r.reader, r.home, r.stale) == (1, 1, 1)
    assert await _tags(db, doc.id) == [("Consensus", "reader"), ("Replication", "shelf")]


async def test_moving_the_shelf_leaves_unscoped_retrieval_alone(scratch_db, monkeypatch):
    """Astra's invariant: a work's visible shelf is browse structure, not evidence."""
    from conftest import _vec

    from library_agent.retrieval import hybrid

    db = scratch_db
    doc = await make_document(db, title="Spanner", body="truetime " * 40)
    await make_document(db, title="Bigtable", body="tablet " * 40)
    _, a, b, _, _ = await _shelves(db)

    async def fixed(q, client=None):
        return _vec(10)

    monkeypatch.setattr(hybrid, "embed_query", fixed)

    async def run():
        hits = await hybrid.hybrid_search(db, "truetime", limit=10, use_lexical=False)
        return [(h.chunk_id, round(h.score, 6)) for h in hits]

    doc.shelf_id = a.id
    await shelving._finish_place(db, doc, a)
    before = await run()
    doc.shelf_id = b.id
    await shelving._finish_place(db, doc, b)
    assert await run() == before
    assert uuid.UUID(str(before[0][0]))  # it found something
