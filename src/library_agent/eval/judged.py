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

import collections
import hashlib
import json
import logging
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

log = logging.getLogger(__name__)

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
    # "human" or "model": who made the marks above. Model-judged cases are scored apart.
    judge: str = "human"
    # Every passage the judge was shown, so agreement can count the ones left unmarked.
    pooled: list[str] = field(default_factory=list)
    # What the model suggested for this case, kept beside a human's marks for agreement:
    # {"supporting": [...], "distractors": [...], "model": "...", "reasons": {id: "..."}}
    model_marks: dict = field(default_factory=dict)

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
    by_model: dict[str, dict] = {}
    by_judge: dict[str, dict] = {}
    for name in names:
        for judge_ in sorted({c.judge for c in cases}):
            into = (
                summary
                if judge_ == "human"
                else by_model
                if judge_ == "model"
                else by_judge.setdefault(judge_, {})
            )
            mine = [c for c in cases if c.judge == judge_]
            for group, rows in _groups(mine, per_case, name):
                into.setdefault(name, {})[group] = _summarise(rows)
    out = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": split or "all",
        "cases": sum(c.judge == "human" for c in cases),
        "model_cases": sum(c.judge == "model" for c in cases),
        "reviewer_cases": sum(c.judge not in ("human", "model") for c in cases),
        "seconds": round(time.time() - started, 1),
        "snapshot": snapshot,
        # The owner's cases; the model's are scored apart until agreement earns them a place.
        "summary": summary,
        "summary_model_judged": by_model,
        # Each outside reviewer's cases, scored apart as well (e.g. "astra").
        "summary_by_judge": by_judge,
        "agreement": agreement(),
        "per_case": per_case,
    }
    path = eval_dir() / "runs" / f"{out['at'].replace(':', '')}-{out['split']}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    path.chmod(0o600)
    out["file"] = str(path)
    return out


def _groups(cases: list[Case], per_case: dict, name: str):
    for group in ["all", *sorted({c.type for c in cases})]:
        rows = [per_case[c.id][name] for c in cases if group == "all" or c.type == group]
        if rows:
            yield group, rows


def _summarise(rows: list[dict]) -> dict:
    recalls = [r["recall@10"] for r in rows if r["recall@10"] is not None]
    return {
        "n": len(rows),
        "mrr": round(sum(r["rr"] for r in rows) / len(rows), 3),
        **{f"hit@{k}": round(sum(r[f"hit@{k}"] for r in rows) / len(rows), 3) for k in KS},
        "recall@10": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "distractors@5": round(sum(r["distractors@5"] for r in rows) / len(rows), 2),
    }


def main() -> None:
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(description="Score the hand-judged set (judge it at /eval).")
    ap.add_argument("split", nargs="?", default="tune", choices=["tune", "test", "all"])
    a = ap.parse_args()
    out = asyncio.run(run(None if a.split == "all" else a.split))
    print(
        f"{out['cases']} yours, {out['model_cases']} model-judged, {out.get('reviewer_cases', 0)} by "
        f"reviewers; {out['seconds']}s -> {out['file']}"
    )
    tables = {"yours": out["summary"], "model-judged": out["summary_model_judged"]}
    tables.update({f"by {j}": t for j, t in (out.get("summary_by_judge") or {}).items()})
    for who, summary in tables.items():
        if not summary:
            continue
        print(f"  {who}:")
        for setup, groups in summary.items():
            g = groups.get("all", {})
            print(
                f"    {setup:24} mrr {g.get('mrr')}  hit@1 {g.get('hit@1')}  hit@5 {g.get('hit@5')}"
                f"  recall@10 {g.get('recall@10')}  distractors@5 {g.get('distractors@5')}"
            )


# ----------------------------------------------------------------- the model as judge

SUGGEST_SYSTEM = """You judge passages for a research library's evaluation set. For a question
and a numbered list of passages, decide for EACH passage whether it supports an answer.

- "supports": the passage itself states something that answers the question, or a
  necessary part of it. Partial support of a multi-part question counts; a passage that
  only shares words or the general topic does not.
- "looks_relevant": on the same topic or sharing the question's terms, but it does not
  answer it -- a neighbouring mechanism, a different protocol, product or version, the
  general case when the question asks a specific one, or a control working as intended
  where the question asks about a failure.
- "unrelated": neither.

Judge only from the passage text. Do not reward a passage for what it might say elsewhere.
Give each verdict a short reason."""

SUGGEST_PROMPT = """Question: {question}

Passages:
{passages}

Return `verdicts`: one per passage, by its number."""

SUGGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supports", "looks_relevant", "unrelated"],
                    },
                    "reason": {"type": "string", "maxLength": 240},
                },
                "required": ["n", "verdict"],
            },
        }
    },
    "required": ["verdicts"],
}

SUGGEST_BATCH = 8  # passages per call: enough context, short enough to judge each one
SUGGEST_CHARS = 1400


def judge_model() -> str:
    from library_agent.llm import providers

    return providers.model_for("chat_general")


def _listed(k: int, p: dict) -> str:
    where = f", p.{p['page']}" if p.get("page") else ""
    body = " ".join(str(p.get("text", "")).split())[:SUGGEST_CHARS]
    return f"[{k}] {p.get('title', '')}{where}\n{body}"


async def suggest(client, question: str, passages: list[dict], model: str | None = None) -> dict:
    """The model's marks for a pooled list: supporting and distractor chunk ids, and a
    reason for each passage it marked. Passages it could not judge are left unmarked."""
    model = model or judge_model()
    marks: dict = {"supporting": [], "distractors": [], "reasons": {}, "model": model}
    for i in range(0, len(passages), SUGGEST_BATCH):
        batch = passages[i : i + SUGGEST_BATCH]
        listing = "\n\n".join(_listed(k, p) for k, p in enumerate(batch, start=1))
        try:
            out = await client.structured(
                model,
                SUGGEST_PROMPT.format(question=question, passages=listing),
                SUGGEST_SCHEMA,
                system=SUGGEST_SYSTEM,
                temperature=0.0,
                seed=1,
                think=True,
                num_predict=4000,
            )
        except Exception:
            log.warning("suggest: a batch of %d went unjudged", len(batch), exc_info=True)
            continue
        for v in out.get("verdicts") or []:
            k = v.get("n")
            if not isinstance(k, int) or not 1 <= k <= len(batch):
                continue
            cid = batch[k - 1]["chunk_id"]
            verdict = v.get("verdict")
            if verdict == "supports":
                marks["supporting"].append(cid)
            elif verdict == "looks_relevant":
                marks["distractors"].append(cid)
            else:
                continue
            reason = " ".join(str(v.get("reason") or "").split())[:240]
            if reason:
                marks["reasons"][cid] = reason
    marks["supporting"] = list(dict.fromkeys(marks["supporting"]))
    marks["distractors"] = [
        x for x in dict.fromkeys(marks["distractors"]) if x not in marks["supporting"]
    ]
    return marks


async def auto_judge(n: int = 10, *, type_: str = "concept") -> list[Case]:
    """Judge the next `n` seed questions with no one watching: pool, suggest, save as
    judge="model". Their type is a default the owner can correct; they are scored apart
    from the owner's cases until agreement says they can be trusted."""
    from library_agent.llm.client import LLM

    made: list[Case] = []
    todo = (await seeds(limit=400))[:n]
    async with LLM() as client:
        for sd in todo:
            passages = await pool(sd["question"])
            marks = await suggest(client, sd["question"], passages)
            case = Case(
                question=sd["question"],
                type=type_ if marks["supporting"] else "unanswerable",
                supporting=marks["supporting"],
                distractors=marks["distractors"],
                source=sd["source"],
                judge="model",
                pooled=[p["chunk_id"] for p in passages],
                model_marks=marks,
                notes="judged by the model, unattended",
            )
            made.append(await judge(case))
    return made


def agreement(cases: list[Case] | None = None) -> dict:
    """How often the model agrees with the owner, on cases the owner judged and the model
    also marked, per question type. Per passage, over everything that was pooled:
    agreement, and the model's precision and recall on "supports", with Cohen's kappa so
    agreement a coin would reach does not count."""
    cases = [
        c
        for c in (cases if cases is not None else load())
        if c.judge == "human" and c.judged and c.model_marks and c.pooled
    ]

    def stats(group: list[Case]) -> dict:
        tp = fp = fn = tn = 0
        for c in group:
            human, model = set(c.supporting), set(c.model_marks.get("supporting") or [])
            for cid in c.pooled:
                h, m = cid in human, cid in model
                tp += h and m
                fp += m and not h
                fn += h and not m
                tn += not h and not m
        total = tp + fp + fn + tn
        if not total:
            return {"cases": len(group), "passages": 0}
        po = (tp + tn) / total
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
        return {
            "cases": len(group),
            "passages": total,
            "agreement": round(po, 3),
            "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        }

    out = {"all": stats(cases)}
    for t in sorted({c.type for c in cases}):
        out[t] = stats([c for c in cases if c.type == t])
    return out


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------- sharing


async def export_bundle() -> dict:
    """The judged set as a reviewer can read it without the library: every case with the
    text of the passages it marked (title, page, section), the owner's marks beside the
    model's (with its reasons), the agreement table, and the latest run's scores. It quotes
    the library -- share it only with someone who may read those passages."""
    cases = [c for c in load() if c.judged]
    ids = {
        x
        for c in cases
        for x in (
            *c.supporting,
            *c.distractors,
            *(c.model_marks.get("supporting") or []),
            *(c.model_marks.get("distractors") or []),
        )
    }
    passages: dict[str, dict] = {}
    if ids:
        async with session_scope() as db:
            rows = (
                await db.execute(
                    text(
                        "select c.id, d.title, s.path, c.page_start, c.text from chunk c"
                        " join document d on d.id = c.document_id"
                        " left join section s on s.id = c.section_id where c.id = any(:ids)"
                    ),
                    {"ids": [uuid.UUID(x) for x in ids]},
                )
            ).all()
        for cid, title, path, page, body in rows:
            passages[str(cid)] = {
                "title": title,
                "section": path,
                "page": page,
                "text": " ".join(body.split())[:1500],
            }
    runs = sorted((eval_dir() / "runs").glob("*.json"))
    latest = json.loads(runs[-1].read_text()) if runs else None
    if latest:
        latest.pop("per_case", None)

    def side(xs: list[str]) -> list[dict]:
        return [{"id": x, **passages.get(x, {"missing": True})} for x in xs]

    return {
        "library_eval": 1,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "Hand-judged questions for The Library's retrieval. 'supporting' passages answer "
        "the question; 'distractors' look relevant but do not. judge='model' cases were marked by "
        "the local model unattended and have not been checked by the owner.",
        "counts": {
            "cases": len(cases),
            "by_judge": dict(collections.Counter(c.judge for c in cases)),
            "by_type": dict(collections.Counter(c.type for c in cases)),
            "by_split": dict(collections.Counter(c.split for c in cases)),
        },
        "agreement": agreement(cases),
        "latest_run": latest,
        "cases": [
            {
                "id": c.id,
                "question": c.question,
                "type": c.type,
                "judge": c.judge,
                "split": c.split,
                "source": c.source,
                "answerable": c.answerable,
                "judged_at": c.judged_at,
                "notes": c.notes,
                "supporting": side(c.supporting),
                "distractors": side(c.distractors),
                "pooled": len(c.pooled),
                "model": {
                    "model": c.model_marks.get("model"),
                    "supporting": c.model_marks.get("supporting") or [],
                    "distractors": c.model_marks.get("distractors") or [],
                    "reasons": c.model_marks.get("reasons") or {},
                }
                if c.model_marks
                else None,
            }
            for c in cases
        ],
    }


# ------------------------------------------------------------------- a reviewer judges

PACKET_INSTRUCTIONS = """You are judging search results for a private research library.

For each item there is a question someone actually asked the library, and the passages its
search setups returned for it, numbered P1, P2, ... Judge each passage on its own text:

- supporting: the passage itself states something that answers the question, or a
  necessary part of it (partial support of a multi-part question counts).
- distractor: on the same topic or sharing the question's words, but it does not answer it
  -- a neighbouring mechanism, a different protocol, product or version, the general case
  when the question asks a specific one, or a control working as intended when the question
  asks about a failure.
- leave the rest unmarked.

Also give the question a type -- one of: exact, concept, mechanism, multi_hop, synthesis,
negation, conflict -- or mark it unanswerable if no passage supports an answer. Notes are
optional (one line: anything the library should know about the question or the results).

Answer with JSON exactly in the `answer_format` shape, one entry per item, using the item's
id and the passage labels (P1, P2, ...)."""


async def export_packet(n: int = 40) -> dict:
    """Questions actually asked, not yet judged, each with its pooled passages as text --
    for someone outside the library to judge. Labels P1.. map back to passages on import."""
    items = []
    for sd in (await seeds(limit=400))[: max(1, min(200, n))]:
        passages = await pool(sd["question"])
        if not passages:
            continue
        items.append(
            {
                "id": uuid.uuid5(uuid.NAMESPACE_URL, sd["question"]).hex[:12],
                "question": sd["question"],
                "source": sd["source"],
                "passages": [
                    {
                        "label": f"P{i}",
                        "chunk_id": p["chunk_id"],
                        "title": p["title"],
                        "section": p["section"],
                        "page": p["page"],
                        "text": " ".join(str(p["text"]).split())[:2000],
                    }
                    for i, p in enumerate(passages, 1)
                ],
            }
        )
    return {
        "library_judging_packet": 1,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "instructions": PACKET_INSTRUCTIONS,
        "types": [t for t in TYPES if t != "unanswerable"],
        "answer_format": {
            "judge": "astra",
            "judgements": [
                {
                    "id": "the item's id",
                    "type": "concept",
                    "unanswerable": False,
                    "supporting": ["P1", "P4"],
                    "distractors": ["P2"],
                    "notes": "",
                }
            ],
        },
        "items": items,
    }


async def import_judgements(packet: dict, answers: dict) -> dict:
    """A reviewer's answers to a packet, saved as cases judged by that reviewer. Labels
    are mapped back to passages through the packet; an item or label the packet does not
    have is skipped and counted, never guessed."""
    judge_name = str(answers.get("judge") or "reviewer").strip().lower()[:20] or "reviewer"
    if judge_name in ("human", "model"):
        judge_name = f"{judge_name}-reviewer"
    by_id = {it["id"]: it for it in packet.get("items") or [] if isinstance(it, dict)}
    saved = skipped = bad_labels = 0
    existing = {c.question.strip().lower(): c for c in load()}
    for j in answers.get("judgements") or []:
        it = by_id.get(str(j.get("id")))
        if not it:
            skipped += 1
            continue
        labels = {p["label"]: p["chunk_id"] for p in it["passages"]}

        def pick(xs, labels=labels) -> list[str]:
            nonlocal bad_labels
            out = []
            for x in xs or []:
                if str(x) in labels:
                    out.append(labels[str(x)])
                else:
                    bad_labels += 1
            return out

        sup, dis = pick(j.get("supporting")), pick(j.get("distractors"))
        t = str(j.get("type") or "concept")
        if j.get("unanswerable") or not sup:
            t, sup = "unanswerable", []
        if t not in TYPES:
            t = "concept"
        prev = existing.get(it["question"].strip().lower())
        if prev and prev.judge == "human":
            skipped += 1  # the owner's own judgement stands
            continue
        case = Case(
            question=it["question"],
            type=t,
            supporting=sup,
            distractors=[x for x in dis if x not in sup],
            source=it.get("source", "ask"),
            notes=str(j.get("notes") or "")[:500],
            judge=judge_name,
            pooled=list(labels.values()),
            **({"id": prev.id} if prev else {}),
        )
        await judge(case)
        saved += 1
    return {"judge": judge_name, "saved": saved, "skipped": skipped, "unknown_labels": bad_labels}
