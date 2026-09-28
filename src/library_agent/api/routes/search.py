"""Raw retrieval endpoint. Chat sits on top of this in Phase 4."""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Query

from library_agent import classification
from library_agent.api.schemas import SearchHitOut, SearchResponse
from library_agent.db.session import SessionDep
from library_agent.library.shelving import expand_category_ids
from library_agent.retrieval.lift import holding_sentence, lift, lift_kind
from library_agent.retrieval.literal import exact_first, literal_hits, literal_terms
from library_agent.retrieval.pipeline import FIND_RETRIEVAL, RetrievalConfig, retrieve

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(min_length=2)],
    db: SessionDep,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    rerank: Annotated[bool, Query()] = True,
    router: Annotated[bool, Query()] = False,
    categories: Annotated[str | None, Query(description="comma-separated category ids")] = None,
    cartridges: Annotated[str | None, Query(description="comma-separated cartridge ids")] = None,
    documents: Annotated[str | None, Query(description="comma-separated document ids")] = None,
    ceiling: Annotated[str | None, Query(description="the highest classification level")] = None,
) -> SearchResponse:
    started = time.time()
    cat_ids = [uuid.UUID(x) for x in categories.split(",") if x.strip()] if categories else None
    if cat_ids:
        cat_ids = await expand_category_ids(db, cat_ids)
    cart_ids = [uuid.UUID(x) for x in cartridges.split(",") if x.strip()] if cartridges else None
    doc_ids = [uuid.UUID(x) for x in documents.split(",") if x.strip()] if documents else None
    terms = literal_terms(q)
    scale = classification.load()
    levels = classification.allowed_levels(ceiling or None, s=scale)
    hits = await retrieve(
        db,
        q,
        limit=limit + 10 if terms else limit,
        category_ids=cat_ids,
        cartridge_ids=cart_ids,
        document_ids=doc_ids,
        levels=levels,
        config=FIND_RETRIEVAL
        if rerank and not router
        else RetrievalConfig(name="api", use_reranker=rerank, use_router=router),
    )
    exact: set = set()
    if terms:
        # A query that names something exactly: passages holding the name come first.
        literal = await literal_hits(
            db,
            terms,
            limit=limit,
            category_ids=cat_ids,
            cartridge_ids=cart_ids,
            document_ids=doc_ids,
            levels=levels,
        )
        hits, exact = exact_first(hits, literal, terms, limit)
    hit_levels = await classification.hit_levels(db, hits, scale)
    lifts = await lift(q, [h.text for h in hits]) if hits else []
    kinds: list[str | None] = []
    for i, h in enumerate(hits):
        holding = None
        if h.chunk_id in exact:
            # The sentence that holds the name, not the one nearest in meaning.
            holding = holding_sentence(h.text, terms)
        if holding:
            lifts[i] = holding
            kinds.append("exact")
        else:
            kinds.append(lift_kind(q, lifts[i]))
    return SearchResponse(
        query=q,
        hits=[
            SearchHitOut(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                document_title=h.document_title,
                section_path=h.section_path,
                page=h.page,
                text=h.text,
                score=h.score,
                dense_rank=h.dense_rank,
                lexical_rank=h.lexical_rank,
                lift=lifted,
                lift_kind=kind,
                exact=h.chunk_id in exact,
                level=lv,
            )
            for h, lifted, kind, lv in zip(hits, lifts, kinds, hit_levels, strict=True)
        ],
        elapsed_seconds=round(time.time() - started, 3),
    )
