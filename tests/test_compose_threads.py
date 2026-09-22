"""Compose plans from the library's threads: cross-document clusters nearest the brief,
their disagreements carried along, scoped like everything else."""

from __future__ import annotations

import uuid

from conftest import _vec, make_document
from sqlalchemy import select

from library_agent.db.models import (
    Artifact,
    ArtifactKind,
    Cluster,
    ClusterMember,
    Embedding,
    OwnerKind,
    Section,
    TargetKind,
)
from library_agent.retrieval import threads


async def _cluster(db, label, docs, *, vec_seed, contradiction=None, summary="A shared theme."):
    cl = Cluster(
        id=uuid.uuid4(),
        label=label,
        size=len(docs) * 2,
        document_count=len(docs),
        has_contradiction=bool(contradiction),
    )
    db.add(cl)
    await db.flush()
    art = Artifact(
        kind=ArtifactKind.CLUSTER_SUMMARY,
        target_kind=TargetKind.CLUSTER,
        target_id=cl.id,
        text=summary,
        model="test",
        prompt_version="v1",
    )
    db.add(art)
    await db.flush()
    db.add(
        Embedding(
            owner_kind=OwnerKind.ARTIFACT,
            owner_id=art.id,
            model="bge-m3",
            vec=_vec(vec_seed),
        )
    )
    # A member is one section's summary artifact; take each document's first.
    for d in docs:
        member_art = (
            await db.execute(
                select(Artifact.id)
                .join(Section, Section.id == Artifact.target_id)
                .where(
                    Section.document_id == d.id,
                    Artifact.kind == ArtifactKind.SECTION_SUMMARY,
                )
                .limit(1)
            )
        ).scalar_one()
        db.add(ClusterMember(cluster_id=cl.id, artifact_id=member_art, document_id=d.id))
    if contradiction:
        db.add(
            Artifact(
                kind=ArtifactKind.CONTRADICTION,
                target_kind=TargetKind.CLUSTER,
                target_id=cl.id,
                text="They disagree on the remedy.",
                data={
                    "claim_a": contradiction[0],
                    "source_a": docs[0].title,
                    "claim_b": contradiction[1],
                    "source_b": docs[1].title,
                },
                model="test",
                prompt_version="v1",
            )
        )
    await db.flush()
    return cl


async def test_threads_nearest_the_brief_with_their_disagreement(scratch_db, monkeypatch):
    db = scratch_db
    a = await make_document(db, title="Volume A", body="a " * 60)
    b = await make_document(db, title="Volume B", body="b " * 60)
    c = await make_document(db, title="Volume C", body="c " * 60)
    await _cluster(
        db,
        "Caching strategies",
        [a, b],
        vec_seed=100,
        contradiction=("write-through is safest", "write-back is faster and safe enough"),
        summary="How the works cache.",
    )
    await _cluster(db, "Unrelated theme", [b, c], vec_seed=900, summary="Something else.")
    # a cluster inside a single document is not a thread
    await _cluster(db, "Just A", [a], vec_seed=101)

    async def near_first(text, client=None):
        return _vec(100)

    monkeypatch.setattr(threads, "embed_query", near_first)
    got = await threads.relevant_threads(db, "caching", limit=5)
    labels = [t["label"] for t in got]
    assert "Caching strategies" in labels and "Just A" not in labels
    assert got[0]["label"] == "Caching strategies"
    top = got[0]
    assert set(top["documents"]) == {"Volume A", "Volume B"} and top["document_count"] == 2
    assert top["contradiction"]["a"]["claim"] == "write-through is safest"
    assert top["contradiction"]["b"]["source"] == "Volume B"

    block = threads.render_threads(got)
    assert "Caching strategies" in block and "⚡" in block and "write-back is faster" in block
    # a thread with no contradiction adds no ⚡ line of its own
    assert block.count("⚡") == 1


async def test_threads_respect_scope(scratch_db, monkeypatch):
    db = scratch_db
    from library_agent.library import taxonomy

    a = await make_document(db, title="Shelved Vol", body="a " * 60)
    b = await make_document(db, title="Other Vol", body="b " * 60)
    cat = await taxonomy.get_or_create(db, "Networking")
    await taxonomy.assign_document(db, a.id, ["Networking"])
    await _cluster(db, "In scope", [a, b], vec_seed=100)

    async def near(text, client=None):
        return _vec(100)

    monkeypatch.setattr(threads, "embed_query", near)
    assert await threads.relevant_threads(db, "q", limit=5, category_ids=[cat.id])
    other = uuid.uuid4()
    assert await threads.relevant_threads(db, "q", limit=5, category_ids=[other]) == []


def test_render_is_empty_without_threads():
    assert threads.render_threads([]) == ""
