"""Raw retrieval endpoint. Chat sits on top of this in Phase 4."""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Query

from library_agent.api.schemas import SearchHitOut, SearchResponse
from library_agent.db.session import SessionDep
from library_agent.library.shelving import expand_category_ids
from library_agent.retrieval.pipeline import RetrievalConfig, retrieve

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(min_length=2)],
    db: SessionDep,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    rerank: Annotated[bool, Query()] = False,
    router: Annotated[bool, Query()] = False,
    categories: Annotated[str | None, Query(description="comma-separated category ids")] = None,
    cartridges: Annotated[str | None, Query(description="comma-separated cartridge ids")] = None,
) -> SearchResponse:
    started = time.time()
    cat_ids = [uuid.UUID(x) for x in categories.split(",") if x.strip()] if categories else None
    if cat_ids:
        cat_ids = await expand_category_ids(db, cat_ids)
    cart_ids = [uuid.UUID(x) for x in cartridges.split(",") if x.strip()] if cartridges else None
    hits = await retrieve(
        db,
        q,
        limit=limit,
        category_ids=cat_ids,
        cartridge_ids=cart_ids,
        config=RetrievalConfig(name="api", use_reranker=rerank, use_router=router),
    )
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
            )
            for h in hits
        ],
        elapsed_seconds=round(time.time() - started, 3),
    )
