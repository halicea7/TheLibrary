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


async def _claims(db, cl, rows):
    """cluster_claim rows: (document, claim text)."""
    from library_agent.db.models import ClusterClaim, ClusterMember
    from library_agent.library.cluster import _hash

    members = {
        m.document_id: m.artifact_id
        for m in (
            await db.execute(select(ClusterMember).where(ClusterMember.cluster_id == cl.id))
        ).scalars()
    }
    for i, (d, claim) in enumerate(rows):
        db.add(
            ClusterClaim(
                cluster_id=cl.id,
                artifact_id=members[d.id],
                claim_index=i,
                claim_hash=_hash(claim),
                document_id=d.id,
                text=claim,
            )
        )
    await db.flush()


async def test_threads_respect_scope(scratch_db, monkeypatch):
    """Scope selects claims, not clusters: a thread is a thread in scope only when two of
    its volumes are there, and it is shown as what those volumes claim -- never with the
    library-wide summary that speaks for volumes outside it."""
    db = scratch_db
    from library_agent.library import taxonomy

    a = await make_document(db, title="Shelved Vol", body="a " * 60)
    b = await make_document(db, title="Other Vol", body="b " * 60)
    c = await make_document(db, title="Third Vol", body="c " * 60)
    cat = await taxonomy.get_or_create(db, "Networking")
    await taxonomy.assign_document(db, a.id, ["Networking"])
    await taxonomy.assign_document(db, c.id, ["Networking"])
    cl = await _cluster(
        db, "Across three", [a, b, c], vec_seed=100, summary="Other Vol settles it."
    )
    await _claims(db, cl, [(a, "A says x"), (b, "B says y"), (c, "C says z")])
    lonely = await _cluster(db, "One in scope", [a, b], vec_seed=100)
    await _claims(db, lonely, [(a, "A alone"), (b, "B elsewhere")])

    async def near(text, client=None):
        return _vec(100)

    monkeypatch.setattr(threads, "embed_query", near)
    got = await threads.relevant_threads(db, "q", limit=5, category_ids=[cat.id])
    assert [t["label"] for t in got] == ["Across three"]  # one in-scope volume: not a thread
    t = got[0]
    assert t["documents"] == ["Shelved Vol", "Third Vol"]
    assert t["scoped"] == {"documents": 2, "of_documents": 3}
    assert {x["claim"] for x in t["claims"]} == {"A says x", "C says z"}
    block = threads.render_threads(got)
    assert "B says y" not in block and "Other Vol settles it" not in block
    assert "2 of its 3 volumes are in scope" in block
    other = uuid.uuid4()
    assert await threads.relevant_threads(db, "q", limit=5, category_ids=[other]) == []
    # unscoped, both threads, each with its summary
    assert len(await threads.relevant_threads(db, "q", limit=5)) == 2


def test_render_is_empty_without_threads():
    assert threads.render_threads([]) == ""


def test_a_disagreement_names_its_side_loosely():
    from library_agent.api.routes.library import same_source

    assert same_source("Nginx", "Nginx — Nginx security vulnerabilities and misconfigurations")
    assert same_source(
        "Smali - Decompiling/[Modifying]/Compiling",
        "Smali - Decompiling/[Modifying]/Compilin — Smali code modification",
    )
    assert not same_source("Nginx", "Upgrade Header Smuggling — HTTP request smuggling")
    assert not same_source("SQL", "SQLmap — tool")  # a short name must match whole


async def test_hold_both_opens_the_claims_own_passage(scratch_db):
    """Evidence for a claim comes from its own section, and within it the passage nearest
    the claim's kept vector -- not a fresh search across the thread's volumes."""
    from library_agent.api.routes.library import evidence
    from library_agent.db.models import Chunk, ClaimVector, ClusterMember, Embedding
    from library_agent.library.cluster import _hash

    db = scratch_db
    a = await make_document(db, title="Volume A", body="a " * 60)
    b = await make_document(db, title="Volume B", body="b " * 60)
    cl = await _cluster(db, "Remedy", [a, b], vec_seed=100)
    await _claims(db, cl, [(a, "Patch first"), (b, "Isolate first")])
    # Volume B's member section gets a second passage, the one the claim is about.
    member = (
        await db.execute(
            select(ClusterMember).where(
                ClusterMember.cluster_id == cl.id, ClusterMember.document_id == b.id
            )
        )
    ).scalar_one()
    sec_id = (await db.get(Artifact, member.artifact_id)).target_id
    extra = Chunk(
        id=uuid.uuid4(),
        document_id=b.id,
        section_id=sec_id,
        order_index=9,
        text="Isolate the host before anything else.",
        page_start=7,
    )
    db.add(extra)
    await db.flush()
    from library_agent.config import settings

    m = settings().embed_model
    db.add(Embedding(owner_kind=OwnerKind.CHUNK, owner_id=extra.id, model=m, vec=_vec(99)))
    db.add(ClaimVector(hash=_hash("Isolate first"), model=m, vec=_vec(99)))
    await db.flush()

    got = await evidence(db, cl.id, "Volume B — what it is about", "Isolate first")
    assert got["chunk_id"] == str(extra.id) and got["page"] == 7 and got["exact"]
    other = await evidence(db, cl.id, "Volume A", "Patch first")
    assert other["document_title"] == "Volume A" and not other["exact"]  # no kept vector
