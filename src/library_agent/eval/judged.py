"""A private, hand-judged evaluation set.

The generated suite (eval/generate.py) asks whether the chunk a question was written
from comes back -- cheap, but it cannot say whether a result set supports a real
question. This set is built from questions actually put to the library (Ask history,
Write briefs, or typed in), each judged by its owner: which passages support an answer,
which only look relevant, or that the library cannot answer it.

Judging is pooled and blind. For a question, several retrieval setups each contribute
their top passages; the judge sees the union in a fixed order with no hint of which setup
found what, so the judgement cannot favour one of them.

The set is a file, not a table: `~/.library-agent/eval/judged.jsonl`, outside the
repository (it quotes the library) and frozen by copying it. Each case is split by the
volumes its evidence comes from, so tuning on "tune" never sees a "test" volume.

Runs are scored per question type and per case, and saved beside the set with what they
ran against (volume and passage counts, embedding model, prompt versions)."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select, text

from library_agent.config import settings
from library_agent.db.models import Chunk, Document
from library_agent.db.session import session_scope
from library_agent.retrieval.literal import exact_first, literal_hits, literal_terms
from library_agent.retrieval.pipeline import LADDER, RetrievalConfig, retrieve

TYPES = (
    "exact",  # names something exactly: an identifier, a command, a number
    "concept",  # what the library says about a topic
    "mechanism",  # how or why something works, or two mechanisms compared
    "multi_hop",  # needs two or more passages that each hold part
    "synthesis",  # across the whole collection or a whole work
    "negation",  # what does NOT hold, or a control that worked as intended
    "conflict",  # the sources disagree
    "unanswerable",  # the library does not hold it
)

# The setups whose top passages are pooled for judging, and scored in a run.
POOL = {
    "dense": next(c for c in LADDER if c.name == "dense_only"),
    "lexical": next(c for c in LADDER if c.name == "lexical_only"),
    "hybrid": next(c for c in LADDER if c.name == "hybrid_rrf"),
    "hybrid_rerank_diverse": next(c for c in LADDER if c.name == "hybrid_rerank_diverse"),
}
POOL_DEPTH = 8
KS = (1, 3, 5, 10)


def eval_dir() -> Path:
    d = settings().storage_dir.parent / "eval"
    (d / "runs").mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)  # it quotes the library
    return d


def set_path() -> Path:
    return eval_dir() / "judged.jsonl"


@dataclass
class Case:
    question: str
    type: str = "concept"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Passages that support an answer (the gold), and ones that look relevant but do not.
    supporting: list[str] = field(default_factory=list)
    distractors: list[str] = field(default_factory=list)
    answerable: bool = True
    source: str = "typed"  # "ask", "write", "typed"
    notes: str = ""
    split: str = ""  # "tune" or "test", by the volumes of its evidence
    judged_at: str = ""

    @property
    def judged(self) -> bool:
        return bool(self.judged_at)


def load() -> list[Case]:
    p = set_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Case(**{k: v for k, v in d.items() if k in Case.__dataclass_fields__}))
    return out


def save(cases: list[Case]) -> None:
    p = set_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(asdict(c)) + "\n" for c in cases), encoding="utf-8")
    tmp.replace(p)
    p.chmod(0o600)


def volume_side(document_id: str) -> str:
    """Each volume's fixed side of the split, from its id."""
    return "test" if int(hashlib.sha256(document_id.encode()).hexdigest(), 16) % 2 else "tune"


def split_for(document_ids: list[str]) -> str:
    """A case is "tune" only when every volume its evidence is in is a tune volume, so
    whatever is tuned on the tune cases has never seen a test volume. (A case drawn from
    both sides counts as test; test may share volumes with tune, never the reverse.)"""
    if not document_ids:
        return "test"
    return "tune" if all(volume_side(d) == "tune" for d in document_ids) else "test"


async def seeds(limit: int = 200) -> list[dict]:
    """Questions actually put to the library, newest first, not already in the set."""
    have = {c.question.strip().lower() for c in load()}
    out: list[dict] = []
    async with session_scope() as db:
        rows = (
            await db.execute(
                text(
                    "select content, max(created_at) as at from message where role = 'user'"
                    " and length(content) between 8 and 600 group by content"
                    " order by at desc limit :n"
                ),
                {"n": limit},
            )
        ).all()
    for content, at in rows:
        if content.strip().lower() not in have:
            out.append({"question": content.strip(), "source": "ask", "at": at.isoformat()})
    from library_agent.chat.compose import list_saved

    for s in list_saved():
        t = s["title"]
        if t.strip().lower() not in have:
            out.append({"question": t, "source": "write", "at": s["modified"]})
    return out


async def pool(question: str) -> list[dict]:
    """The union of every pooled setup's top passages, in a fixed order that does not
    reveal which setup found which (by volume, then page)."""
    from library_agent.llm.client import LLM

    found: dict = {}
    async with LLM() as client, session_scope() as db:
        for cfg in POOL.values():
            for h in await retrieve(db, question, config=cfg, client=client, limit=POOL_DEPTH):
                found.setdefault(h.chunk_id, h)
        for h in await literal_hits(db, literal_terms(question), limit=POOL_DEPTH):
            found.setdefault(h.chunk_id, h)
    hits = sorted(found.values(), key=lambda h: (h.document_title.lower(), h.page or 0))
    return [
        {
            "chunk_id": str(h.chunk_id),
            "document_id": str(h.document_id),
            "title": h.document_title,
            "section": h.section_path,
            "page": h.page,
            "text": h.text,
        }
        for h in hits
    ]


async def judge(case: Case) -> Case:
    """Record a judgement: stamp it and split it by the volumes of its evidence."""
    ids = [uuid.UUID(x) for x in case.supporting]
    docs: list[str] = []
    if ids:
        async with session_scope() as db:
            docs = [
                str(d)
                for d in (
                    await db.execute(select(Chunk.document_id).where(Chunk.id.in_(ids)))
                ).scalars()
            ]
    case.split = split_for(docs)
    case.answerable = bool(case.supporting) and case.type != "unanswerable"
    if not case.answerable:
        case.type = "unanswerable"
    case.judged_at = datetime.now(UTC).isoformat(timespec="seconds")
    cases = [c for c in load() if c.id != case.id] + [case]
    save(cases)
    return case


def score(ranked: list[str], supporting: set[str], distractors: set[str]) -> dict:
    """One case under one setup: where the first supporting passage came, how many of the
    supporting ones were found by 10, and how many distractors took a top-5 place."""
    first = next((i for i, c in enumerate(ranked, 1) if c in supporting), None)
    return {
        "first": first,
        "rr": 1.0 / first if first else 0.0,
        **{f"hit@{k}": bool(first and first <= k) for k in KS},
        "recall@10": len(supporting & set(ranked[:10])) / len(supporting) if supporting else None,
        "distractors@5": len(distractors & set(ranked[:5])),
    }


async def _run_setup(db, client, case: Case, name: str, cfg: RetrievalConfig) -> list[str]:
    if name == "find":  # what the Find tab does: ranked, then exact names first
        terms = literal_terms(case.question)
        hits = await retrieve(
            db, case.question, client=client, limit=20, config=RetrievalConfig(name="api")
        )
        if terms:
            lit = await literal_hits(db, terms, limit=10)
            hits, _ = exact_first(hits, lit, terms, 10)
        return [str(h.chunk_id) for h in hits[:10]]
    hits = await retrieve(db, case.question, config=cfg, client=client, limit=10)
    return [str(h.chunk_id) for h in hits]


async def run(split: str | None = "tune", setups: list[str] | None = None) -> dict:
    """Score every judged, answerable case in `split` (None: all) under each setup. The
    result is saved to eval/runs/ and returned."""
    from library_agent.llm.client import LLM

    cases = [c for c in load() if c.judged and c.answerable and (split is None or c.split == split)]
    names = setups or ["find", *POOL]
    started = time.time()
    per_case: dict[str, dict] = {}
    async with LLM() as client, session_scope() as db:
        for c in cases:
            per_case[c.id] = {"question": c.question, "type": c.type}
            for name in names:
                ranked = await _run_setup(db, client, c, name, POOL.get(name, RetrievalConfig()))
                per_case[c.id][name] = score(ranked, set(c.supporting), set(c.distractors))
        snapshot = {
            "volumes": (await db.execute(select(func.count()).select_from(Document))).scalar(),
            "passages": (await db.execute(select(func.count()).select_from(Chunk))).scalar(),
            "embed_model": settings().embed_model,
            "prompt_versions": settings().prompt_versions,
        }
    summary: dict[str, dict] = {}
    for name in names:
        for group in ["all", *sorted({c.type for c in cases})]:
            rows = [per_case[c.id][name] for c in cases if group == "all" or c.type == group]
            if not rows:
                continue
            recalls = [r["recall@10"] for r in rows if r["recall@10"] is not None]
            summary.setdefault(name, {})[group] = {
                "n": len(rows),
                "mrr": round(sum(r["rr"] for r in rows) / len(rows), 3),
                **{f"hit@{k}": round(sum(r[f"hit@{k}"] for r in rows) / len(rows), 3) for k in KS},
                "recall@10": round(sum(recalls) / len(recalls), 3) if recalls else None,
                "distractors@5": round(sum(r["distractors@5"] for r in rows) / len(rows), 2),
            }
    out = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": split or "all",
        "cases": len(cases),
        "seconds": round(time.time() - started, 1),
        "snapshot": snapshot,
        "summary": summary,
        "per_case": per_case,
    }
    path = eval_dir() / "runs" / f"{out['at'].replace(':', '')}-{out['split']}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    path.chmod(0o600)
    out["file"] = str(path)
    return out


def main() -> None:
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(description="Score the hand-judged set (judge it at /eval).")
    ap.add_argument("split", nargs="?", default="tune", choices=["tune", "test", "all"])
    a = ap.parse_args()
    out = asyncio.run(run(None if a.split == "all" else a.split))
    print(f"{out['cases']} cases, {out['seconds']}s -> {out['file']}")
    for setup, groups in out["summary"].items():
        g = groups.get("all", {})
        print(
            f"  {setup:24} mrr {g.get('mrr')}  hit@5 {g.get('hit@5')}  recall@10 {g.get('recall@10')}"
        )


if __name__ == "__main__":
    main()
