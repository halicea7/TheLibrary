"""The hand-judged evaluation set: seed it, pool and judge a question, run it.

See eval/judged.py. Loopback or token only, like the rest of the API: the set quotes the
library."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from library_agent.db.models import Chunk, Document
from library_agent.db.session import SessionDep
from library_agent.eval import judged
from library_agent.llm.client import LLM

router = APIRouter(prefix="/api/eval", tags=["eval"])


class CaseIn(BaseModel):
    question: str
    type: str = "concept"
    id: str | None = None
    supporting: list[str] = []
    distractors: list[str] = []
    source: str = "typed"
    notes: str = ""
    pooled: list[str] = []
    model_marks: dict = {}


class SuggestIn(BaseModel):
    question: str
    chunk_ids: list[str]


@router.get("/cases")
async def cases() -> dict:
    cs = judged.load()
    return {
        "types": list(judged.TYPES),
        "cases": [judged.asdict(c) for c in cs],
        "judged": sum(c.judged for c in cs),
        "by_type": {t: sum(c.type == t for c in cs if c.judged) for t in judged.TYPES},
        "by_split": {s: sum(c.split == s for c in cs if c.judged) for s in ("tune", "test")},
        "by_judge": {j: sum(c.judge == j for c in cs if c.judged) for j in ("human", "model")},
        "file": str(judged.set_path()),
    }


@router.get("/seeds")
async def seeds(limit: int = 200) -> list[dict]:
    return await judged.seeds(limit)


@router.get("/pool")
async def pool(q: str) -> list[dict]:
    if len(q.strip()) < 3:
        raise HTTPException(422, "the question is too short")
    return await judged.pool(q)


@router.post("/cases")
async def judge(c: CaseIn) -> dict:
    if c.type not in judged.TYPES:
        raise HTTPException(422, f"type must be one of {', '.join(judged.TYPES)}")
    case = judged.Case(
        question=c.question.strip(),
        type=c.type,
        supporting=c.supporting,
        distractors=[x for x in c.distractors if x not in c.supporting],
        source=c.source,
        notes=c.notes,
        judge="human",  # saved from the page: the owner confirmed or corrected the marks
        pooled=c.pooled,
        model_marks=c.model_marks,
        **({"id": c.id} if c.id else {}),
    )
    return judged.asdict(await judged.judge(case))


@router.delete("/cases/{case_id}")
async def forget(case_id: str) -> dict:
    cs = judged.load()
    judged.save([c for c in cs if c.id != case_id])
    return {"removed": len(cs) - len(judged.load())}


@router.post("/run")
async def run(split: str = "tune") -> dict:
    if split not in ("tune", "test", "all"):
        raise HTTPException(422, "split is tune, test or all")
    out = await judged.run(None if split == "all" else split)
    out.pop("per_case", None)
    return out


def _model_ready() -> None:
    from library_agent.llm.liveness import liveness

    if not liveness.alive:
        raise HTTPException(503, f"the model is not answering: {liveness.detail}")


@router.post("/suggest")
async def suggest(body: SuggestIn, db: SessionDep) -> dict:
    """The model's marks for passages already gathered, for the owner to confirm or
    correct. Their text is read from the library, not taken from the request."""
    _model_ready()
    ids = [uuid.UUID(x) for x in body.chunk_ids[:40]]
    rows = {
        str(c.id): (c, title)
        for c, title in (
            await db.execute(
                select(Chunk, Document.title)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.id.in_(ids))
            )
        ).all()
    }
    passages = [
        {
            "chunk_id": cid,
            "title": rows[cid][1],
            "page": rows[cid][0].page_start,
            "text": rows[cid][0].text,
        }
        for cid in body.chunk_ids
        if cid in rows
    ]
    async with LLM() as client:
        return await judged.suggest(client, body.question, passages)


@router.post("/auto")
async def auto(n: int = 10) -> dict:
    """Judge the next `n` asked questions unattended. Saved as model-judged: scored apart
    from the owner's cases, and shown for correction."""
    _model_ready()
    made = await judged.auto_judge(max(1, min(25, n)))
    return {"judged": len(made), "unanswerable": sum(not c.answerable for c in made)}


@router.get("/agreement")
async def agreement() -> dict:
    return judged.agreement()
