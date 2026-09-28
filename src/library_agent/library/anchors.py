"""Claim anchors: every Tier 1 claim tied to the passage of its own section that says it.

Tier 1 writes claims per section, so a claim knew its section but not the words behind it.
Here each claim's kept vector (claim_vector, by its text's hash) is compared with the
passages of that section, and the nearest is kept with how near it is. No model is
called: the vectors are already in the library.

What the nearness means was set from the library itself (a sample of 1,607 claims:
median 0.65; 5% under 0.50; 14% under 0.55):

    anchored     >= 0.55   the passage says it, or near enough to cite
    weak         0.45-0.55 partial support at best; worth a look
    unsupported  < 0.45    nothing in its own section says it: the reader may have
                           overreached, or merged a claim from elsewhere
    unembedded   no vector yet (a volume read since the last threads rebuild)

Anchors are rebuilt after every threads rebuild, when the claim vectors are fresh."""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import ClaimAnchor, ClaimVector

log = logging.getLogger(__name__)

ANCHORED = 0.55
WEAK = 0.45


def claim_hash(text_: str) -> str:
    return hashlib.sha256(text_.encode()).hexdigest()[:32]


def state_for(similarity: float | None) -> str:
    if similarity is None:
        return "unembedded"
    if similarity >= ANCHORED:
        return "anchored"
    if similarity >= WEAK:
        return "weak"
    return "unsupported"


def _vec(v) -> np.ndarray:
    if isinstance(v, str):  # a raw query hands back pgvector's text form, "[x,y,...]"
        v = np.fromstring(v.strip("[]"), sep=",", dtype=np.float32)
    a = np.asarray(v.to_list() if hasattr(v, "to_list") else v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n else a


@dataclass
class AnchorResult:
    claims: int
    states: dict[str, int]


async def anchor_document(db: AsyncSession, document_id: uuid.UUID) -> Counter:
    """Anchor every claim of one volume's section summaries; replaces what was there."""
    model = settings().embed_model
    arts = (
        await db.execute(
            text(
                "select a.id, a.target_id, a.data->'claims' from artifact a"
                " join section s on s.id = a.target_id"
                " where a.kind = 'section_summary' and a.data ? 'claims' and s.document_id = :d"
            ),
            {"d": document_id},
        )
    ).all()
    await db.execute(delete(ClaimAnchor).where(ClaimAnchor.document_id == document_id))
    if not arts:
        return Counter()
    chunks = (
        await db.execute(
            text(
                "select c.id, c.section_id, e.vec from chunk c"
                " join embedding e on e.owner_kind = 'chunk' and e.owner_id = c.id and e.model = :m"
                " where c.document_id = :d and c.kind = 'text'"
            ),
            {"m": model, "d": document_id},
        )
    ).all()
    by_section: dict = {}
    for cid, sid, vec in chunks:
        by_section.setdefault(sid, ([], []))
        by_section[sid][0].append(cid)
        by_section[sid][1].append(_vec(vec))
    matrices = {sid: (ids, np.stack(vs)) for sid, (ids, vs) in by_section.items()}
    claims = [(aid, sid, i, str(c)) for aid, sid, cl in arts for i, c in enumerate(cl or [])]
    hashes = list({claim_hash(c) for *_, c in claims})
    vecs = {
        h: _vec(v)
        for h, v in (
            await db.execute(
                select(ClaimVector.hash, ClaimVector.vec).where(
                    ClaimVector.hash.in_(hashes), ClaimVector.model == model
                )
            )
        ).all()
    }
    rows, tally = [], Counter()
    for aid, sid, idx, claim in claims:
        h = claim_hash(claim)
        chunk_id, sim = None, None
        q = vecs.get(h)
        m = matrices.get(sid)
        if q is not None and m is not None:
            scores = m[1] @ q
            best = int(np.argmax(scores))
            chunk_id, sim = m[0][best], round(float(scores[best]), 4)
        st = state_for(sim) if (q is not None or m is None) else "unembedded"
        if q is not None and m is None:
            st = "unsupported"  # a claim whose section has no passage at all
        tally[st] += 1
        rows.append(
            {
                "artifact_id": aid,
                "claim_index": idx,
                "claim_hash": h,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "similarity": sim,
                "state": st,
            }
        )
    for i in range(0, len(rows), 2000):
        await db.execute(pg_insert(ClaimAnchor).values(rows[i : i + 2000]).on_conflict_do_nothing())
    return tally


async def build_anchors(db_factory, *, progress=None) -> AnchorResult:
    """Anchor every volume's claims, one transaction per volume. `db_factory` is the
    session_scope context manager (so a long run commits as it goes)."""
    async with db_factory() as db:
        docs = list(
            (
                await db.execute(
                    text(
                        "select distinct s.document_id from artifact a join section s on s.id = a.target_id"
                        " where a.kind = 'section_summary' and a.data ? 'claims'"
                    )
                )
            ).scalars()
        )
    total = Counter()
    for i, did in enumerate(docs):
        async with db_factory() as db:
            total += await anchor_document(db, did)
        if progress and (i % 25 == 0 or i + 1 == len(docs)):
            await progress(i + 1, len(docs), "anchoring claims")
    return AnchorResult(claims=sum(total.values()), states=dict(total))


async def report(db: AsyncSession, *, worst: int = 15) -> dict:
    """Counts by state, per volume where the reader overreached most, and the weakest
    claims with the passage each was nearest -- for a look by eye."""
    states = dict(
        (await db.execute(text("select state, count(*) from claim_anchor group by state"))).all()
    )
    per_doc = (
        await db.execute(
            text(
                "select d.title, count(*) filter (where ca.state = 'unsupported') as unsupported,"
                " count(*) as claims from claim_anchor ca join document d on d.id = ca.document_id"
                " group by d.title having count(*) >= 5"
                " order by (count(*) filter (where ca.state = 'unsupported'))::float / count(*) desc limit 10"
            )
        )
    ).all()
    weakest = (
        await db.execute(
            text(
                "select d.title, a.data->'claims'->>ca.claim_index as claim, ca.similarity,"
                " left(c.text, 240) as passage from claim_anchor ca"
                " join artifact a on a.id = ca.artifact_id join document d on d.id = ca.document_id"
                " left join chunk c on c.id = ca.chunk_id"
                " where ca.similarity is not null order by ca.similarity limit :n"
            ),
            {"n": worst},
        )
    ).all()
    return {
        "states": states,
        "thresholds": {"anchored": ANCHORED, "weak": WEAK},
        "volumes_most_unsupported": [
            {"title": t, "unsupported": u, "claims": n} for t, u, n in per_doc
        ],
        "weakest": [
            {"title": t, "claim": c, "similarity": s, "nearest_passage": p}
            for t, c, s, p in weakest
        ],
    }


def main() -> None:
    import argparse
    import asyncio
    import json

    from library_agent.db.session import session_scope

    ap = argparse.ArgumentParser(description="Tie every Tier 1 claim to its passage.")
    ap.add_argument("--report", action="store_true", help="only report, do not rebuild")
    a = ap.parse_args()

    async def run() -> None:
        if not a.report:
            r = await build_anchors(session_scope)
            print(f"{r.claims} claims: {r.states}")
        async with session_scope() as db:
            print(json.dumps(await report(db), indent=1, ensure_ascii=False, default=str))

    asyncio.run(run())


if __name__ == "__main__":
    main()
