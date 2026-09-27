"""The hand-judged evaluation set: seed it, pool and judge a question, run it.

See eval/judged.py. Loopback or token only, like the rest of the API: the set quotes the
library."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from library_agent.eval import judged

router = APIRouter(prefix="/api/eval", tags=["eval"])


class CaseIn(BaseModel):
    question: str
    type: str = "concept"
    id: str | None = None
    supporting: list[str] = []
    distractors: list[str] = []
    source: str = "typed"
    notes: str = ""


@router.get("/cases")
async def cases() -> dict:
    cs = judged.load()
    return {
        "types": list(judged.TYPES),
        "cases": [judged.asdict(c) for c in cs],
        "judged": sum(c.judged for c in cs),
        "by_type": {t: sum(c.type == t for c in cs if c.judged) for t in judged.TYPES},
        "by_split": {s: sum(c.split == s for c in cs if c.judged) for s in ("tune", "test")},
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
