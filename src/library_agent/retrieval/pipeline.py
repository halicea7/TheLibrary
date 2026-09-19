"""The retrieval pipeline: route → fuse → rerank.

Every stage is switchable so the eval harness can attribute recall changes to a specific
component rather than to the pipeline as a whole."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.llm import rerank as rerank_mod
from library_agent.llm.embed import embed_query
from library_agent.llm.ollama import Ollama
from library_agent.retrieval.hybrid import SearchHit, hybrid_search
from library_agent.retrieval.router import route


@dataclass(frozen=True)
class RetrievalConfig:
    name: str = "default"
    use_dense: bool = True
    use_lexical: bool = True
    use_router: bool = False
    use_reranker: bool = False
    # Include Tier 2 reflection vectors, mapped back to their chunk.
    use_reflections: bool = False
    # Candidates actually scored by the cross-encoder. The first stage already
    # places the answer near the top, so scoring the full pool buys little and
    # costs ~30ms per pair.
    rerank_depth: int = 20
    # Postgres ts_rank_cd has no IDF term, so an unweighted lexical half injects
    # noise from common words. Down-weighting it is what keeps fusion a net gain.
    weight_dense: float | None = None
    weight_lexical: float | None = None
    router_top_documents: int | None = None
    # Diversity: at most this many passages from one document until the limit is filled,
    # then the rest in rank order. A comparative question otherwise gets five passages from
    # the one volume that scored highest and nothing from the other side.
    per_document: int | None = None
    # Share of the candidate budget searched corpus-wide even when routing, so a router
    # miss degrades results instead of losing the answer entirely.
    router_global_reserve: float | None = None
    candidate_pool: int | None = None
    final_top_k: int | None = None


async def retrieve(
    db: AsyncSession,
    query: str,
    *,
    config: RetrievalConfig | None = None,
    client: Ollama | None = None,
    limit: int | None = None,
    category_ids: list[uuid.UUID] | None = None,
    cartridge_ids: list[uuid.UUID] | None = None,
) -> list[SearchHit]:
    cfg = settings()
    rc = config or RetrievalConfig()
    top_k = limit or rc.final_top_k or cfg.final_top_k
    pool = rc.candidate_pool or cfg.candidate_pool
    reserve = (
        rc.router_global_reserve
        if rc.router_global_reserve is not None
        else cfg.router_global_reserve
    )

    qvec = await embed_query(query, client)

    routed_docs: list[uuid.UUID] | None = None
    if rc.use_router:
        hits = await route(db, qvec, top_k=rc.router_top_documents or cfg.router_top_documents)
        routed_docs = [d for d, _ in hits] or None

    common = {
        "query_vector": qvec,
        "use_dense": rc.use_dense,
        "use_lexical": rc.use_lexical,
        "client": client,
        "weight_dense": rc.weight_dense if rc.weight_dense is not None else cfg.weight_dense,
        "use_reflections": rc.use_reflections,
        "category_ids": category_ids,
        "cartridge_ids": cartridge_ids,
        "weight_lexical": rc.weight_lexical
        if rc.weight_lexical is not None
        else cfg.weight_lexical,
    }

    if routed_docs:
        # Retrieve wide inside the routed set, then top up from the whole corpus.
        routed_pool = max(1, int(pool * (1 - reserve)))
        global_pool = max(1, pool - routed_pool)
        primary = await hybrid_search(
            db, query, limit=pool, document_ids=routed_docs, pool=routed_pool, **common
        )
        fallback = await hybrid_search(db, query, limit=global_pool, pool=global_pool, **common)
        seen = {h.chunk_id for h in primary}
        candidates = primary + [h for h in fallback if h.chunk_id not in seen]
    else:
        candidates = await hybrid_search(db, query, limit=pool, pool=pool, **common)

    if rc.use_reranker and candidates and rerank_mod.available():
        # Score only the head. The cross-encoder costs ~65ms per pair on MPS, so
        # reranking the full pool would put an interactive query at ~6s for a result the
        # first stage already ranks highly.
        head, tail = candidates[: rc.rerank_depth], candidates[rc.rerank_depth :]
        scores = rerank_mod.rerank(query, [f"{h.context_prefix}\n\n{h.text}" for h in head])
        for h, s in zip(head, scores, strict=True):
            h.rerank_score = s
        head.sort(key=lambda h: -(h.rerank_score or 0.0))
        # The unscored tail keeps its fusion order, below everything the reranker saw.
        candidates = head + tail

    if rc.per_document:
        candidates = diversify(candidates, per_document=rc.per_document, limit=top_k)
    return candidates[:top_k]


def diversify(hits: list, *, per_document: int, limit: int) -> list:
    """Two passes: first take up to `per_document` from each document in rank order; if
    that leaves room, fill it with what was skipped, still in rank order."""
    taken: dict = {}
    first, rest = [], []
    for h in hits:
        if taken.get(h.document_id, 0) < per_document:
            taken[h.document_id] = taken.get(h.document_id, 0) + 1
            first.append(h)
        else:
            rest.append(h)
    return (first + rest)[:limit]


# Ablation ladder: each rung adds exactly one component.
LADDER = [
    RetrievalConfig(name="dense_only", use_lexical=False),
    RetrievalConfig(name="lexical_only", use_dense=False),
    RetrievalConfig(name="hybrid_rrf"),
    RetrievalConfig(name="hybrid_router", use_router=True),
    RetrievalConfig(name="hybrid_rerank", use_reranker=True),
    RetrievalConfig(name="hybrid_router_rerank", use_router=True, use_reranker=True),
    RetrievalConfig(name="hybrid_rerank_diverse", use_reranker=True, per_document=2),
]

# What chat should use: reranking wins on conversational phrasing.
CHAT_RETRIEVAL = RetrievalConfig(name="chat", use_reranker=True, rerank_depth=20)
# What the raw search box should use: keyword queries are hurt by the cross-encoder.
KEYWORD_RETRIEVAL = RetrievalConfig(name="keyword", use_reranker=False)
