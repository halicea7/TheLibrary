"""Cross-document clustering.

The hierarchy in the spec topped out at the document, but the stated goal is talking to
the *library*. This is the layer above: a theme running across five papers becomes one
retrievable thing.

**Clustering operates on claims, not section summaries.** Measured on a 12-document corpus:
section summaries produced 29 clusters with 2 spanning more than one document, because a
summary like "this section describes RAPTOR's clustering algorithm" is irreducibly about
its own paper. The same corpus's 1041 extracted claims produced 119 clusters with 23
cross-document -- an order of magnitude better, and the themes are real ("bigger models are
better" across four papers, shared optimiser hyperparameters across three).

It is cheap either way: clustering embeddings costs nothing and only the one summary call
per cluster touches the GPU."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import (
    Artifact,
    ArtifactKind,
    Cluster,
    ClusterMember,
    Document,
    Embedding,
    OwnerKind,
    TargetKind,
)
from library_agent.llm.embed import embed_texts
from library_agent.llm.ollama import Ollama
from library_agent.reading import genre as genre_mod

log = logging.getLogger(__name__)

CLUSTER_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["label", "summary"],
}

CLUSTER_SCHEMA["properties"]["significant"] = {"type": "boolean"}
CLUSTER_SCHEMA["required"].append("significant")

CLUSTER_PROMPT = """These claims come from different documents in one personal library.
They were grouped together automatically because they are about the same thing.
{genre_note}
{members}

Write the library's entry for this theme.

- label: 2-5 words naming the theme. Not a document title -- the theme itself.
- summary: 3-6 sentences on what the library as a whole says about this, synthesising
  across sources rather than listing them. Where sources agree, say so; where they differ
  in approach or conclusion, say that too. Refer to sources by document title.
- significant: false if this grouping is incidental rather than a real theme -- shared
  tooling, hardware, or boilerplate ("both used 8 GPUs", "both are on HuggingFace") rather
  than a substantive idea. Be strict; incidental groupings clutter the library."""


@dataclass
class ClusterResult:
    clusters: int
    members: int
    noise: int
    cross_document: int


async def _load_sections(
    db: AsyncSession, client: Ollama
) -> tuple[list[uuid.UUID], list[uuid.UUID], list[str], np.ndarray]:
    """Individual claims extracted at Tier 1, each embedded on its own.

    Claims rather than section summaries: a claim is atomic and comparable across papers,
    whereas a section summary carries so much document-specific detail that it clusters
    with its own siblings. See the module docstring for the measured difference."""
    rows = (
        await db.execute(
            text("""
            select a.id as artifact_id, claim as text, s.document_id
            from artifact a
            join section s on s.id = a.target_id,
            lateral jsonb_array_elements_text(a.data->'claims') as claim
            where a.kind = :kind and a.data ? 'claims'
            """),
            {"kind": ArtifactKind.SECTION_SUMMARY.value},
        )
    ).all()
    if not rows:
        return [], [], [], np.zeros((0, 0))

    texts = [r.text for r in rows]
    vecs = np.array(await embed_texts(texts, client), dtype=np.float32)
    # Normalise so euclidean distance is monotonic in cosine distance.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-9, None)
    return [r.artifact_id for r in rows], [r.document_id for r in rows], texts, vecs


def _cluster(vectors: np.ndarray, min_cluster_size: int, method: str = "leaf") -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    model = HDBSCAN(
        min_cluster_size=max(2, min_cluster_size),
        min_samples=1,
        metric="euclidean",  # vectors are L2-normalised above
        copy=True,
        cluster_selection_method=method,
    )
    return model.fit_predict(vectors)


async def build_clusters(
    db: AsyncSession,
    *,
    client: Ollama | None = None,
    min_cluster_size: int = 3,
    method: str = "leaf",
    summarise: bool = True,
    progress=None,
    gate=None,
) -> ClusterResult:
    cfg = settings()
    own = client is None
    c = client or Ollama()
    try:
        artifact_ids, doc_ids, texts, vectors = await _load_sections(db, c)
        if len(artifact_ids) < 6:
            raise ValueError(f"only {len(artifact_ids)} section summaries; run Tier 1 first")

        labels = _cluster(vectors, min_cluster_size, method)
        groups: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            if lab >= 0:
                groups.setdefault(int(lab), []).append(i)

        docs_all = list((await db.execute(select(Document))).scalars())
        titles = {d.id: d.title for d in docs_all}
        genres = {d.id: d.genre for d in docs_all}

        # Clusters are derived state -- rebuild wholesale rather than reconciling.
        await db.execute(
            delete(Embedding).where(
                Embedding.owner_kind == OwnerKind.ARTIFACT,
                Embedding.owner_id.in_(
                    select(Artifact.id).where(Artifact.kind == ArtifactKind.CLUSTER_SUMMARY)
                ),
            )
        )
        await db.execute(delete(Artifact).where(Artifact.kind == ArtifactKind.CLUSTER_SUMMARY))
        await db.execute(delete(Cluster))
        await db.flush()

        cross_doc = 0
        to_embed: list[tuple[uuid.UUID, str]] = []

        for n, (_lab, idxs) in enumerate(sorted(groups.items())):
            if gate:
                await gate()
            docs_in = {doc_ids[i] for i in idxs}
            cluster = Cluster(method="hdbscan", size=len(idxs), document_count=len(docs_in))
            db.add(cluster)
            await db.flush()
            seen_artifacts: set[uuid.UUID] = set()
            for i in idxs:
                # (cluster_id, artifact_id) is the PK, and one section can contribute
                # several claims to the same cluster.
                if artifact_ids[i] in seen_artifacts:
                    continue
                seen_artifacts.add(artifact_ids[i])
                db.add(
                    ClusterMember(
                        cluster_id=cluster.id,
                        artifact_id=artifact_ids[i],
                        document_id=doc_ids[i],
                    )
                )
            if len(docs_in) > 1:
                cross_doc += 1

            if summarise:
                members = "\n".join(
                    f"- [{titles.get(doc_ids[i], '?')[:40]}] {texts[i][:220]}" for i in idxs[:16]
                )
                # Phrase the entry for the kind of writing it is: a wiki of runbooks is
                # not a body of findings.
                dom = genre_mod.dominant([genres.get(doc_ids[i]) for i in idxs])
                phrase = genre_mod.ENTRY_BY_GENRE.get(dom or "")
                genre_note = f"Note: {phrase}.\n" if phrase else ""
                try:
                    out = await c.structured(
                        cfg.reader_model,
                        CLUSTER_PROMPT.format(members=members, genre_note=genre_note),
                        CLUSTER_SCHEMA,
                        temperature=0.0,
                        instructions=CLUSTER_PROMPT,
                        seed=1,
                    )
                except Exception:
                    log.warning("cluster summary failed for %s", cluster.id, exc_info=True)
                    continue
                if not out.get("significant", True):
                    # "Not a real theme" is the sampling-dependent call. Confirm it once
                    # with a different seed before hiding the cluster; when the two
                    # disagree, the cluster is shown.
                    try:
                        again = await c.structured(
                            cfg.reader_model,
                            CLUSTER_PROMPT.format(members=members, genre_note=genre_note),
                            CLUSTER_SCHEMA,
                            temperature=0.0,
                            instructions=CLUSTER_PROMPT,
                            seed=2,
                        )
                        if again.get("significant", True):
                            out = again
                    except Exception:  # noqa: BLE001
                        log.debug("second significance vote failed; keeping the first")
                cluster.label = (out.get("label") or "").strip()[:120]
                cluster.has_contradiction = False
                if not out.get("significant", True):
                    # Incidental grouping (shared tooling/hardware). Keep the membership
                    # for the graph, but do not surface or embed it as a library theme.
                    continue
                art = Artifact(
                    kind=ArtifactKind.CLUSTER_SUMMARY,
                    target_kind=TargetKind.CLUSTER,
                    target_id=cluster.id,
                    text=(out.get("summary") or "").strip(),
                    data={
                        **out,
                        "claims": [texts[i] for i in idxs[:16]],
                        # Parallel to claims: who said each, so a conflict can be quoted
                        # as "X says … / Y says …" rather than paraphrased.
                        "claim_sources": [titles.get(doc_ids[i], "?") for i in idxs[:16]],
                        "genre": dom,
                    },
                    model=cfg.reader_model,
                    prompt_version=cfg.prompt_versions.get("cluster_summary", "v1"),
                    tier=1,
                )
                db.add(art)
                await db.flush()
                to_embed.append((art.id, f"{cluster.label}\n\n{art.text}"))
            if progress:
                await progress(n + 1, len(groups))

        if to_embed:
            vecs = await embed_texts([t for _, t in to_embed], c)
            for (aid, _), vec in zip(to_embed, vecs, strict=True):
                db.add(
                    Embedding(
                        owner_kind=OwnerKind.ARTIFACT,
                        owner_id=aid,
                        model=cfg.embed_model,
                        vec=vec,
                    )
                )
        await db.flush()
    finally:
        if own:
            await c.aclose()

    return ClusterResult(
        clusters=len(groups),
        members=int((labels >= 0).sum()),
        noise=int((labels < 0).sum()),
        cross_document=cross_doc,
    )
