"""Contradiction detection across documents.

Only a library-level view can notice that two sources disagree. A per-document RAG system
structurally cannot: it never holds two papers' claims side by side.

Runs over cross-document clusters only -- a cluster drawn from one paper cannot contradict
itself in any interesting way -- so the cost is one call per multi-source theme."""

from __future__ import annotations

import hashlib
import logging
import re

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import Artifact, ArtifactKind, Cluster, TargetKind
from library_agent.llm import providers
from library_agent.llm.client import LLM
from library_agent.llm.ollama import Ollama
from library_agent.reading import genre as genre_mod

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
        "claim_a": {"type": "string", "maxLength": 300},
        "source_a": {"type": "string", "maxLength": 120},
        "claim_b": {"type": "string", "maxLength": 300},
        "source_b": {"type": "string", "maxLength": 120},
        # The one thing both clashing claims are about, named the same way in both.
        # Empty means they are about different things -- and then there is no conflict.
        # Not required: an empty required string trips the placeholder guard, and empty
        # is the honest answer for most clusters.
        "subject": {"type": "string", "maxLength": 160},
        "disagreement": {"type": "boolean"},
    },
    "required": ["analysis", "explanation", "sources", "disagreement"],
}

PROMPT = """These claims come from different documents in one library, on the theme
"{label}". Each claim is prefixed with its document's title and, after a dash, what that
document is about -- the claim's scope.

{claims}

Decide whether these sources genuinely conflict.
{genre_rule}
A claim is about the thing its document is about. "The head node is login01" in a guide to
one cluster and "the head node is cerberus" in a guide to another are two facts about two
systems, not a disagreement. The same goes for different hosts, accounts, tenants, sites,
versions, environments, datasets, hardware, or dates: a claim scoped to one is not
contradicted by a claim scoped to another.

A conflict means the sources cannot both be right about the SAME thing:
- opposite directions for the same measurement of the same method
- incompatible numbers for the same quantity of the same system under the same conditions
- opposite instructions for the same system ("enable X on it" vs "disable X on it")
- one source saying a method works where another, on the same setup, says it does not

NOT a conflict:
- claims scoped to different systems, hosts, accounts, versions, sites or setups
- sources covering different aspects of a topic, or one being more specific
- the same instruction stated as done and as to be done, or differing in wording, tense
  or level of detail
- a requirement in one place and a stricter requirement in another (they nest)
- results that are simply unrelated

Work through it in this order: name what each claim is about, including its scope; find
two claims about the same thing under the same scope; only then check whether they point
in opposite directions.

Return, in this order:
- analysis: brief — one line per claim naming what it is about and its scope, then one
  line on whether any two share a subject and point in opposite directions. Under 140
  words.
- explanation: if they conflict, what each side claims and why they cannot both hold.
  Otherwise one sentence on why the differences are compatible.
- sources: document titles involved in the conflict (empty list if none).
- claim_a, source_a, claim_b, source_b: if they conflict, the two claims that cannot both
  hold -- the claim text copied word for word from the list above, without the bracketed
  document name, which goes in source_a / source_b. Leave all four empty if no conflict.
- subject: the single thing both clashing claims are about, named so that it is plainly
  the same in both (the same host, the same method, the same quantity). Empty if there is
  no such shared subject -- and then there is no conflict.
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


# A "conflict" between pages that each say their content moved elsewhere is the stub
# pages of a mirrored wiki talking, not two sources disagreeing.
_BOILERPLATE = re.compile(
    r"\b(content|page|section)s?\b.{0,40}\b(moved|relocated)\b", re.IGNORECASE
)


def _explanation_is_junk(text: str | None) -> bool:
    if not text or len(text.strip()) < 40:
        return True
    return bool(_INSTRUCTION_SHAPED.match(text)) or bool(_BOILERPLATE.search(text[:300]))


async def judge_cluster(
    c: Ollama,
    model: str,
    label: str | None,
    claims: list,
    titles: list[str],
    claim_sources: list | None = None,
    scopes: dict[str, str] | None = None,
    genre: str | None = None,
) -> dict | None:
    """One cluster's verdict by majority. Returns None if no vote came back. `scopes`
    maps a document title to its orientation line, so the judge sees what each claim's
    document is about -- the difference between two hosts and a disagreement."""
    srcs = list(claim_sources or [])
    scopes = scopes or {}

    def tag(i: int) -> str:
        if i >= len(srcs):
            return ""
        title = str(srcs[i])
        scope = scopes.get(title, "")
        return f"[{title[:40]} — {scope[:90]}] " if scope else f"[{title[:40]}] "

    body = "\n".join(f"- {tag(i)}{str(x)[:220]}" for i, x in enumerate(claims[:16]))
    rule = genre_mod.CONFLICT_BY_GENRE.get(genre or "")
    genre_rule = f"\n{rule}\n" if rule else ""
    body += "\n\nDocuments involved: " + ", ".join(titles)
    # Identical runs used to return 0, 1, 3, 4 and 7 findings at temperature 0.1: which
    # borderline pairs fire is a coin flip. So each cluster is judged up to three times
    # at temperature 0 with different seeds and a finding needs a majority; a first
    # clean "no" ends it early, since most clusters are clean.
    votes: list[dict] = []
    for seed in (1, 2, 3):
        try:
            out = await c.structured(
                model,
                PROMPT.format(label=label or "(unlabelled)", claims=body, genre_rule=genre_rule),
                SCHEMA,
                temperature=0.0,
                instructions=PROMPT,
                seed=seed,
            )
        except Exception:
            log.warning("contradiction check failed for cluster %r", label, exc_info=True)
            continue
        if out.get("disagreement") and _explanation_is_junk(out.get("explanation")):
            # No usable explanation means no usable verdict.
            out["disagreement"] = False
        if out.get("disagreement") and _explanation_denies(out.get("explanation")):
            # The model can still answer true while its own reasoning says no --
            # "the claims are not in conflict because..." with disagreement=true was
            # observed even with the verdict field last. The reasoning wins.
            out["disagreement"] = False
        if out.get("disagreement") and len(str(out.get("subject") or "").strip()) < 3:
            # No shared subject named means two facts about two things.
            out["disagreement"] = False
        votes.append(out)
        yes = sum(1 for v in votes if v.get("disagreement"))
        no = len(votes) - yes
        if yes >= 2 or no >= 2:
            break
    if not votes:
        return None
    yes_votes = [v for v in votes if v.get("disagreement")]
    out = yes_votes[0] if len(yes_votes) >= 2 else votes[0]
    out["disagreement"] = len(yes_votes) >= 2
    out["votes"] = f"{len(yes_votes)}/{len(votes)}"
    if out["disagreement"]:
        # A quoted claim is shown as a quote, so it must be one: keep the pair only if
        # both halves are found among the claims the model was given.
        pool = " ".join(str(x) for x in claims).lower()
        for k in ("claim_a", "claim_b"):
            q = " ".join(str(out.get(k) or "").split())
            # The model tends to copy the "[Title]" prefix along with the claim; peel it
            # off and let it stand as the source when none was given.
            m = re.match(r"^\[([^\]]{1,80})\]\s*(.+)$", q)
            if m:
                q = m.group(2)
                out.setdefault(k.replace("claim", "source"), "")
                if not out.get(k.replace("claim", "source")):
                    out[k.replace("claim", "source")] = m.group(1)
            out[k] = q
            if len(q) < 15 or q.lower()[:60] not in pool:
                out["claim_a"] = out["claim_b"] = ""
                break
    return out


async def find_contradictions(
    db: AsyncSession, *, client: Ollama | None = None, progress=None, gate=None
) -> dict[str, int]:
    cfg = settings()
    rows = (
        await db.execute(
            text("""
            select c.id, c.label, c.member_key, c.judged_key, c.has_contradiction,
                   a.data->'claims' as claims,
                   a.data->'claim_sources' as claim_sources,
                   a.data->>'genre' as genre,
                   array_agg(distinct d.title) as titles,
                   jsonb_object_agg(d.title, coalesce(o.data->>'one_liner', '')) as scopes
            from cluster c
            join artifact a on a.target_id = c.id and a.kind = :ck
            join cluster_member m on m.cluster_id = c.id
            join document d on d.id = m.document_id
            left join artifact o on o.target_id = d.id and o.kind = :ok
            where c.document_count > 1
            group by c.id, c.label, c.member_key, c.judged_key, c.has_contradiction, a.data
            """),
            {"ck": ArtifactKind.CLUSTER_SUMMARY.value, "ok": ArtifactKind.ORIENTATION.value},
        )
    ).all()

    # A verdict stands while the cluster it was reached on does: the same claims, judged
    # by the same model under the same prompt. Only the rest are judged, and for those
    # the pass is authoritative -- without the reset a verdict flipped to false on re-run
    # left the previous true standing, and an instruction-echo "contradiction" survived
    # two fixes that way.
    model = providers.model_for("threads")
    version = cfg.prompt_versions.get("contradiction", "v1")

    def stamp(row) -> str:
        return hashlib.sha256(f"{row.member_key}:{model}:{version}".encode()).hexdigest()[:32]

    todo = [r for r in rows if r.member_key is None or r.judged_key != stamp(r)]
    kept = len(rows) - len(todo)
    if todo:
        ids = [r.id for r in todo]
        await db.execute(update(Cluster).where(Cluster.id.in_(ids)).values(has_contradiction=False))
        await db.execute(
            text("delete from artifact where kind = :k and target_id = any(:ids)"),
            {"k": ArtifactKind.CONTRADICTION.value, "ids": ids},
        )
    phase = f"{len(todo):,} changed themes, {kept:,} kept" if kept else None

    own = client is None
    c = client or LLM()
    found = sum(1 for r in rows if r.id not in {x.id for x in todo} and r.has_contradiction)
    try:
        for n, row in enumerate(todo):
            if gate:
                await gate()
            claims = row.claims or []
            if len(claims) < 2:
                await db.execute(
                    update(Cluster).where(Cluster.id == row.id).values(judged_key=stamp(row))
                )
                continue
            out = await judge_cluster(
                c,
                providers.model_for("threads"),
                row.label,
                claims,
                row.titles,
                row.claim_sources,
                row.scopes,
                row.genre,
            )
            if out is None:
                continue  # the judge failed; left unstamped, so the next pass returns to it
            await db.execute(
                update(Cluster).where(Cluster.id == row.id).values(judged_key=stamp(row))
            )
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
                            model=providers.model_for("threads"),
                            prompt_version=cfg.prompt_versions.get("contradiction", "v1"),
                            tier=1,
                        )
                    )
            if progress:
                await progress(n + 1, len(todo), phase)
        await db.flush()
    finally:
        if own:
            await c.aclose()
    return {"clusters_checked": len(todo), "kept": kept, "contradictions": found}


async def list_contradictions(db: AsyncSession) -> list[dict]:
    rows = (
        await db.execute(
            text("""
            select c.id, c.label, a.text, a.data->'sources' as sources,
                   a.data->>'claim_a' as claim_a, a.data->>'source_a' as source_a,
                   a.data->>'claim_b' as claim_b, a.data->>'source_b' as source_b,
                   c.document_count
            from cluster c
            join artifact a on a.target_id = c.id and a.kind = :k
            where c.has_contradiction
            order by c.document_count desc
            """),
            {"k": ArtifactKind.CONTRADICTION.value},
        )
    ).all()
    return [
        {
            "cluster_id": str(r.id),
            "label": r.label,
            "explanation": r.text,
            "sources": r.sources,
            "document_count": r.document_count,
            "pair": (
                [
                    {"source": r.source_a or "", "claim": r.claim_a},
                    {"source": r.source_b or "", "claim": r.claim_b},
                ]
                if r.claim_a and r.claim_b
                else None
            ),
        }
        for r in rows
    ]
