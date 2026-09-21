"""Threads rebuild incrementally: a cluster with the same claims as before is kept with
its summary and verdict; claim vectors are embedded once."""

from __future__ import annotations

import hashlib

import numpy as np
from conftest import make_document
from sqlalchemy import select, update

from library_agent.db.models import Artifact, ArtifactKind, ClaimVector, Cluster
from library_agent.library.cluster import build_clusters
from library_agent.library.contradictions import find_contradictions

THEMES = {"cache": 0, "kernel": 1, "tunnel": 2}


class FakeClient:
    """Deterministic 'embeddings' (claims about the same thing land together) and a
    summary per call, counting what it was asked."""

    def __init__(self):
        self.embedded = 0
        self.summaries = 0
        self.judged = 0

    async def embed(self, texts, model=None):
        self.embedded += len(texts)
        out = []
        for t in texts:
            v = np.zeros(1024, dtype=np.float32)
            v[THEMES.get(t.split()[0].split("-")[0], 7)] = 1.0  # first word picks the direction
            v[8 + int(hashlib.md5(t.encode()).hexdigest(), 16) % 500] = 0.05  # distinct points
            out.append(v.tolist())
        return out

    async def structured(self, model, prompt, schema, **kw):
        if "disagreement" in schema.get("properties", {}):
            self.judged += 1
            return {"disagreement": False, "explanation": "they agree", "subject": "x"}
        self.summaries += 1
        return {
            "label": "A theme",
            "summary": "What the library says about it.",
            "significant": True,
        }

    async def aclose(self):
        pass


async def _seed(db, monkeypatch):
    # Three themes of five claims each, spread over two documents.
    a = await make_document(db, title="Alpha", body="alpha " * 60)
    b = await make_document(db, title="Beta", body="beta " * 60)
    secs = list(
        (
            await db.execute(select(Artifact).where(Artifact.kind == ArtifactKind.SECTION_SUMMARY))
        ).scalars()
    )
    words = ["cache", "kernel", "tunnel"]
    for n, art in enumerate(secs):
        art.data = {"claims": [f"{w} claim {n} {i}" for w in words for i in range(2)]}
    await db.flush()

    from library_agent.library import cluster as cl

    async def fake_embed(texts, client=None):
        return await client.embed(texts)

    monkeypatch.setattr(cl, "embed_texts", fake_embed)
    return a, b


async def test_second_rebuild_costs_nothing(scratch_db, monkeypatch):
    db = scratch_db
    await _seed(db, monkeypatch)
    c = FakeClient()
    r1 = await build_clusters(db, client=c, min_cluster_size=3)
    assert r1.clusters >= 2 and c.summaries == r1.clusters
    first = {x.id: x.member_key for x in (await db.execute(select(Cluster))).scalars()}
    assert all(first.values())
    embedded_once = c.embedded
    assert (await db.execute(select(ClaimVector))).scalars().all()

    await find_contradictions(db, client=c)
    judged_once = c.judged
    assert judged_once >= 1

    r2 = await build_clusters(db, client=c, min_cluster_size=3)
    assert r2.clusters == r1.clusters
    # nothing re-embedded beyond the summaries' own vectors, nothing re-summarised
    assert c.summaries == r1.clusters
    assert c.embedded == embedded_once
    second = {x.id: x.member_key for x in (await db.execute(select(Cluster))).scalars()}
    assert second == first, "the same clusters, same rows"
    out = await find_contradictions(db, client=c)
    assert c.judged == judged_once and out["kept"] == out["kept"] and out["clusters_checked"] == 0


async def test_changed_claims_redo_only_their_cluster(scratch_db, monkeypatch):
    db = scratch_db
    await _seed(db, monkeypatch)
    c = FakeClient()
    r1 = await build_clusters(db, client=c, min_cluster_size=3)
    await find_contradictions(db, client=c)
    before_s, before_j = c.summaries, c.judged
    # One theme's claims are reworded (a volume re-read); the other themes stand.
    art = (
        (await db.execute(select(Artifact).where(Artifact.kind == ArtifactKind.SECTION_SUMMARY)))
        .scalars()
        .first()
    )
    art.data = {
        "claims": [
            x.replace("tunnel", "tunnel-rewritten") if "tunnel" in x else x
            for x in art.data["claims"]
        ]
    }
    await db.flush()
    r2 = await build_clusters(db, client=c, min_cluster_size=3)
    assert r2.clusters == r1.clusters
    redone = c.summaries - before_s
    assert 1 <= redone < r1.clusters
    out = await find_contradictions(db, client=c)
    assert c.judged > before_j and 0 < out["clusters_checked"] <= redone


async def test_new_prompt_version_redoes_everything(scratch_db, monkeypatch):
    db = scratch_db
    await _seed(db, monkeypatch)
    c = FakeClient()
    r1 = await build_clusters(db, client=c, min_cluster_size=3)
    await db.execute(
        update(Artifact)
        .where(Artifact.kind == ArtifactKind.CLUSTER_SUMMARY)
        .values(prompt_version="v0")
    )
    await build_clusters(db, client=c, min_cluster_size=3)
    assert c.summaries == 2 * r1.clusters
