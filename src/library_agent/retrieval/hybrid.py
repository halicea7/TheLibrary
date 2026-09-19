"""Hybrid retrieval: dense + lexical, fused with Reciprocal Rank Fusion.

Both halves run in one SQL statement against one store. Dense alone misses exact terms
(method names, acronyms, author surnames tokenize into subwords that embeddings blur);
lexical alone misses paraphrase. RRF needs no score calibration between them, which
matters because cosine similarity and ts_rank_cd are not on comparable scales."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import OwnerKind
from library_agent.llm.embed import embed_query
from library_agent.llm.ollama import Ollama


@dataclass
class SearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    section_path: str
    page: int | None
    text: str
    score: float
    dense_rank: int | None
    lexical_rank: int | None
    context_prefix: str = ""
    rerank_score: float | None = None

    def citation(self) -> str:
        loc = f", p.{self.page}" if self.page else ""
        return f"{self.document_title}{loc}"


def _build_sql(use_dense: bool, use_lexical: bool, use_reflections: bool = False) -> str:
    """Compose the fusion query. Ablating either half is what lets the eval attribute
    recall gains to dense vs lexical rather than guessing."""
    ctes, score_terms, joins, conds = [], [], [], []
    if use_dense:
        # Tier 2 reflections are separate vectors that point back at their chunk, so a
        # reflection hit counts as finding its passage. Taking the best score per chunk
        # keeps a chunk from occupying two candidate slots.
        source = (
            """(
        select e.owner_id as chunk_id, e.vec
        from embedding e join chunk c on c.id = e.owner_id
        where e.owner_kind = 'chunk' and e.model = :emodel
          and (cast(:docs as uuid[]) is null or c.document_id = any(cast(:docs as uuid[])))
      and (cast(:cats as uuid[]) is null or exists (
          select 1 from document_category dc
          where dc.document_id = c.document_id
            and dc.category_id = any(cast(:cats as uuid[]))
      ) or exists (
          select 1 from chunk_category cc
          where cc.chunk_id = c.id and cc.category_id = any(cast(:cats as uuid[]))
      ))
      and (cast(:carts as uuid[]) is null or exists (
          select 1 from cartridge_document cd
          where cd.document_id = c.document_id
            and cd.cartridge_id = any(cast(:carts as uuid[]))
      ))

        union all
        select a.target_id as chunk_id, e.vec
        from embedding e
        join artifact a on a.id = e.owner_id and a.kind = 'reflection'
        join chunk c on c.id = a.target_id
        where e.owner_kind = 'artifact' and e.model = :emodel
          and (cast(:docs as uuid[]) is null or c.document_id = any(cast(:docs as uuid[])))
      and (cast(:cats as uuid[]) is null or exists (
          select 1 from document_category dc
          where dc.document_id = c.document_id
            and dc.category_id = any(cast(:cats as uuid[]))
      ) or exists (
          select 1 from chunk_category cc
          where cc.chunk_id = c.id and cc.category_id = any(cast(:cats as uuid[]))
      ))
      and (cast(:carts as uuid[]) is null or exists (
          select 1 from cartridge_document cd
          where cd.document_id = c.document_id
            and cd.cartridge_id = any(cast(:carts as uuid[]))
      ))

    )"""
            if use_reflections
            else """(
        select e.owner_id as chunk_id, e.vec
        from embedding e join chunk c on c.id = e.owner_id
        where e.owner_kind = :okind and e.model = :emodel
          and (cast(:docs as uuid[]) is null or c.document_id = any(cast(:docs as uuid[])))
      and (cast(:cats as uuid[]) is null or exists (
          select 1 from document_category dc
          where dc.document_id = c.document_id
            and dc.category_id = any(cast(:cats as uuid[]))
      ) or exists (
          select 1 from chunk_category cc
          where cc.chunk_id = c.id and cc.category_id = any(cast(:cats as uuid[]))
      ))
      and (cast(:carts as uuid[]) is null or exists (
          select 1 from cartridge_document cd
          where cd.document_id = c.document_id
            and cd.cartridge_id = any(cast(:carts as uuid[]))
      ))

    )"""
        )
        ctes.append(f"""dense as (
    select chunk_id as id, row_number() over (order by d) as rk
    from (
        select chunk_id, min(vec <=> cast(:qv as halfvec)) as d
        from {source} src
        group by chunk_id
        order by d
        limit :pool
    ) best
)""")
        score_terms.append("coalesce(:w_dense / (:k + dense.rk), 0)")
        joins.append("left join dense on dense.id = c.id")
        conds.append("dense.id is not null")
    if use_lexical:
        ctes.append("""lex as (
    select c.id,
           row_number() over (
               order by ts_rank_cd(c.fts, or_tsquery(:q)) desc
           ) as rk
    from chunk c
    where c.fts @@ or_tsquery(:q)
      and (cast(:docs as uuid[]) is null or c.document_id = any(cast(:docs as uuid[])))
      and (cast(:cats as uuid[]) is null or exists (
          select 1 from document_category dc
          where dc.document_id = c.document_id
            and dc.category_id = any(cast(:cats as uuid[]))
      ) or exists (
          select 1 from chunk_category cc
          where cc.chunk_id = c.id and cc.category_id = any(cast(:cats as uuid[]))
      ))
      and (cast(:carts as uuid[]) is null or exists (
          select 1 from cartridge_document cd
          where cd.document_id = c.document_id
            and cd.cartridge_id = any(cast(:carts as uuid[]))
      ))

    limit :pool
)""")
        score_terms.append("coalesce(:w_lex / (:k + lex.rk), 0)")
        joins.append("left join lex on lex.id = c.id")
        conds.append("lex.id is not null")

    return f"""
with {", ".join(ctes)}
select c.id, c.document_id, d.title, coalesce(s.path, '') as section_path,
       c.page_start, c.text, c.context_prefix,
       {" + ".join(score_terms)} as score,
       {"dense.rk" if use_dense else "null"} as dense_rank,
       {"lex.rk" if use_lexical else "null"} as lexical_rank
from chunk c
join document d on d.id = c.document_id
left join section s on s.id = c.section_id
{" ".join(joins)}
where {" or ".join(conds)}
order by score desc, c.id
limit :limit
"""


async def hybrid_search(
    db: AsyncSession,
    query: str,
    *,
    limit: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    client: Ollama | None = None,
    query_vector: list[float] | None = None,
    use_dense: bool = True,
    use_lexical: bool = True,
    pool: int | None = None,
    weight_dense: float = 1.0,
    weight_lexical: float = 1.0,
    use_reflections: bool = False,
    category_ids: list[uuid.UUID] | None = None,
    cartridge_ids: list[uuid.UUID] | None = None,
) -> list[SearchHit]:
    cfg = settings()
    if not (use_dense or use_lexical):
        raise ValueError("at least one of use_dense/use_lexical must be enabled")
    vec = query_vector if query_vector is not None else await embed_query(query, client)
    params = {
        "qv": str(vec),
        "q": query,
        "okind": OwnerKind.CHUNK.value,
        "emodel": cfg.embed_model,
        "docs": [str(d) for d in document_ids] if document_ids else None,
        "cats": [str(x) for x in category_ids] if category_ids else None,
        # A cartridge is a room: scoping to one keeps every hit inside it.
        "carts": [str(x) for x in cartridge_ids] if cartridge_ids else None,
        "pool": pool or cfg.candidate_pool,
        "k": cfg.rrf_k,
        "w_dense": weight_dense,
        "w_lex": weight_lexical,
        "limit": limit or cfg.final_top_k,
    }
    if use_reflections:
        params.pop("okind", None)
    if not use_dense:
        params.pop("w_dense")
    if not use_lexical:
        params.pop("w_lex")
    rows = (
        await db.execute(text(_build_sql(use_dense, use_lexical, use_reflections)), params)
    ).all()
    return [
        SearchHit(
            chunk_id=r.id,
            document_id=r.document_id,
            document_title=r.title,
            section_path=r.section_path,
            page=r.page_start,
            text=r.text,
            score=float(r.score),
            dense_rank=r.dense_rank,
            lexical_rank=r.lexical_rank,
        )
        for r in rows
    ]
