"""Claim anchors: each Tier 1 claim tied to the passage of its section nearest it."""

from __future__ import annotations

from conftest import _vec, make_document
from sqlalchemy import select

from library_agent.config import settings
from library_agent.db.models import Artifact, ArtifactKind, Chunk, ClaimAnchor, ClaimVector, Section
from library_agent.library.anchors import anchor_document, claim_hash, state_for


def test_states_follow_the_measured_thresholds():
    assert state_for(0.7) == "anchored" and state_for(0.55) == "anchored"
    assert state_for(0.5) == "weak" and state_for(0.44) == "unsupported"
    assert state_for(None) == "unembedded"


async def test_each_claim_is_tied_to_the_passage_that_says_it(scratch_db):
    db = scratch_db
    doc = await make_document(db, title="Raft", body="raft " * 60)
    sec = (
        (
            await db.execute(
                select(Section).where(Section.document_id == doc.id).order_by(Section.order_index)
            )
        )
        .scalars()
        .first()
    )
    chunk = (await db.execute(select(Chunk).where(Chunk.section_id == sec.id))).scalars().first()
    summary = (
        await db.execute(
            select(Artifact).where(
                Artifact.target_id == sec.id, Artifact.kind == ArtifactKind.SECTION_SUMMARY
            )
        )
    ).scalar_one()
    summary.data = {
        "claims": [
            "Leaders are elected by majority",
            "Something the section never says",
            "Not yet embedded",
        ]
    }
    m = settings().embed_model
    # the first claim's vector is the passage's own (make_document gives chunk i the vector _vec(10 + i))
    db.add(ClaimVector(hash=claim_hash("Leaders are elected by majority"), model=m, vec=_vec(10)))
    db.add(ClaimVector(hash=claim_hash("Something the section never says"), model=m, vec=_vec(999)))
    await db.flush()
    tally = await anchor_document(db, doc.id)
    rows = {
        r.claim_index: r
        for r in (
            await db.execute(select(ClaimAnchor).where(ClaimAnchor.artifact_id == summary.id))
        ).scalars()
    }
    assert (
        rows[0].state == "anchored" and rows[0].chunk_id == chunk.id and rows[0].similarity > 0.99
    )
    assert rows[1].state == "unsupported" and rows[1].similarity < 0.45  # a random direction
    assert rows[2].state == "unembedded" and rows[2].chunk_id is None
    assert tally["anchored"] == 1 and tally["unsupported"] == 1
    # anchoring again replaces rather than duplicates
    await anchor_document(db, doc.id)
    n = len(
        (await db.execute(select(ClaimAnchor).where(ClaimAnchor.document_id == doc.id)))
        .scalars()
        .all()
    )
    assert n == 3
