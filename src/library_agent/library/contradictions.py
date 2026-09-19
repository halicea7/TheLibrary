"""Contradiction detection across documents.

Only a library-level view can notice that two sources disagree. A per-document RAG system
structurally cannot: it never holds two papers' claims side by side.

Runs over cross-document clusters only -- a cluster drawn from one paper cannot contradict
itself in any interesting way -- so the cost is one call per multi-source theme."""

from __future__ import annotations

import logging
import re

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import Artifact, ArtifactKind, Cluster, TargetKind
from library_agent.llm.ollama import Ollama

log = logging.getLogger(__name__)

# Field order matters. Constrained generation emits properties in schema order, so a
# verdict placed first is committed to before the model has reasoned. With
# `disagreement` first, the model wrote "they report opposite effects" in the explanation
# while still returning false. Reasoning fields go first; the boolean comes last.
SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "explanation": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
        "disagreement": {"type": "boolean"},
    },
    "required": ["analysis", "explanation", "sources", "disagreement"],
}

PROMPT = """These claims come from different documents in one library, on the theme
"{label}".

{claims}

Decide whether these sources genuinely conflict.

A conflict means the sources cannot both be right about the same thing:
- opposite directions for the same measurement ("X improved recall" vs "X degraded recall")
- incompatible numbers for the same quantity under the same conditions
- opposite recommendations about the same technique ("use X" vs "do not use X")
- one source reporting a method works where another reports it does not

NOT a conflict:
- different setups, datasets, hardware, or model sizes producing different numbers
- sources covering different aspects of a topic
- one source being more specific than another
- results that are simply unrelated

Work through it in this order: identify what quantity or question each claim is about; if
two claims are about the same thing, check whether they point in opposite directions. If
they do, that is a conflict even when the sources study different systems -- say so and
name it.

Return, in this order:
- analysis: brief — one line per claim naming what it is about, then one line on whether
  any two are about the same thing and point in opposite directions. Under 120 words.
- explanation: if they conflict, what each side claims and why they cannot both hold.
  Otherwise one sentence on why the differences are compatible.
- sources: document titles involved in the conflict (empty list if none).
- disagreement: your verdict, following from the analysis above."""


_DENIAL = re.compile(
    r"\b(not|no|never)\s+(in\s+)?(genuine(ly)?\s+|direct(ly)?\s+|actual(ly)?\s+)?"
    r"(conflict|contradict|disagree|incompatible|inconsisten)|"
    r"\b(are|is|remain)\s+(fully\s+|entirely\s+|mutually\s+)?(compatible|consistent|complementary)\b",
    re.IGNORECASE,
)


def _explanation_denies(text: str | None) -> bool:
    """Does the explanation say the sources do *not* conflict?"""
    return bool(text) and bool(_DENIAL.search(text[:400]))


# "Explain why they conflict or why they don't." -- a paraphrase of the instruction rather
# than an answer. The verbatim echo guard cannot see a paraphrase; the shape gives it away.
_INSTRUCTION_SHAPED = re.compile(
    r"^\s*(explain|state|describe|return|provide|give|write|list|say|identify|note)\b",
    re.IGNORECASE,
)


def _explanation_is_junk(text: str | None) -> bool:
    if not text or len(text.strip()) < 40:
        return True
    return bool(_INSTRUCTION_SHAPED.match(text))


async def find_contradictions(
    db: AsyncSession, *, client: Ollama | None = None, progress=None
) -> dict[str, int]:
    cfg = settings()
    rows = (
        await db.execute(
            text("""
            select c.id, c.label, a.data->'claims' as claims,
                   array_agg(distinct d.title) as titles
            from cluster c
            join artifact a on a.target_id = c.id and a.kind = :ck
            join cluster_member m on m.cluster_id = c.id
            join document d on d.id = m.document_id
            where c.document_count > 1
            group by c.id, c.label, a.data
            """),
            {"ck": ArtifactKind.CLUSTER_SUMMARY.value},
        )
    ).all()

    # A pass is authoritative. Without this a verdict flipped to false on re-run left
    # the previous true standing -- an instruction-echo "contradiction" survived two
    # fixes that way.
    await db.execute(update(Cluster).values(has_contradiction=False))
    await db.execute(
        text("delete from artifact where kind = :k"), {"k": ArtifactKind.CONTRADICTION.value}
    )

    own = client is None
    c = client or Ollama()
    found = 0
    try:
        for n, row in enumerate(rows):
            claims = row.claims or []
            if len(claims) < 2:
                continue
            body = "\n".join(f"- {str(x)[:220]}" for x in claims[:16])
            body += "\n\nDocuments involved: " + ", ".join(row.titles)
            # Identical runs used to return 0, 1, 3, 4 and 7 findings at temperature 0.1:
            # which borderline pairs fire is a coin flip. So each cluster is judged up to
            # three times at temperature 0 with different seeds and a finding needs a
            # majority; a first clean "no" ends it early, since most clusters are clean.
            votes: list[dict] = []
            for seed in (1, 2, 3):
                try:
                    out = await c.structured(
                        cfg.reader_model,
                        PROMPT.format(label=row.label or "(unlabelled)", claims=body),
                        SCHEMA,
                        temperature=0.0,
                        instructions=PROMPT,
                        seed=seed,
                    )
                except Exception:
                    log.warning("contradiction check failed for cluster %s", row.id, exc_info=True)
                    continue
                if out.get("disagreement") and _explanation_is_junk(out.get("explanation")):
                    # No usable explanation means no usable verdict.
                    out["disagreement"] = False
                if out.get("disagreement") and _explanation_denies(out.get("explanation")):
                    # The model can still answer true while its own reasoning says no --
                    # "the claims are not in conflict because..." with disagreement=true was
                    # observed even with the verdict field last. The reasoning wins.
                    out["disagreement"] = False
                votes.append(out)
                yes = sum(1 for v in votes if v.get("disagreement"))
                no = len(votes) - yes
                if yes >= 2 or no >= 2:
                    break
            if not votes:
                continue
            yes_votes = [v for v in votes if v.get("disagreement")]
            out = yes_votes[0] if len(yes_votes) >= 2 else votes[0]
            out["disagreement"] = len(yes_votes) >= 2
            out["votes"] = f"{len(yes_votes)}/{len(votes)}"
            if out.get("disagreement"):
                found += 1
                await db.execute(
                    update(Cluster).where(Cluster.id == row.id).values(has_contradiction=True)
                )
                existing = (
                    await db.execute(
                        select(Artifact).where(
                            Artifact.kind == ArtifactKind.CONTRADICTION,
                            Artifact.target_id == row.id,
                        )
                    )
                ).scalar_one_or_none()
                textval = (out.get("explanation") or "").strip()
                if existing:
                    existing.text = textval
                    existing.data = out
                else:
                    db.add(
                        Artifact(
                            kind=ArtifactKind.CONTRADICTION,
                            target_kind=TargetKind.CLUSTER,
                            target_id=row.id,
                            text=textval,
                            data=out,
                            model=cfg.reader_model,
                            prompt_version=cfg.prompt_versions.get("contradiction", "v1"),
                            tier=1,
                        )
                    )
            if progress:
                await progress(n + 1, len(rows))
        await db.flush()
    finally:
        if own:
            await c.aclose()
    return {"clusters_checked": len(rows), "contradictions": found}


async def list_contradictions(db: AsyncSession) -> list[dict]:
    rows = (
        await db.execute(
            text("""
            select c.id, c.label, a.text, a.data->'sources' as sources
            from cluster c
            join artifact a on a.target_id = c.id and a.kind = :k
            where c.has_contradiction
            order by c.document_count desc
            """),
            {"k": ArtifactKind.CONTRADICTION.value},
        )
    ).all()
    return [
        {"cluster_id": str(r.id), "label": r.label, "explanation": r.text, "sources": r.sources}
        for r in rows
    ]
