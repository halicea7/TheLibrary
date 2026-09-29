"""The benchmark's questions, drawn from the library it runs in.

A benchmark with fixed questions only works on the shelf they were written for. So
`./library bench init` samples passages across the volumes you hold -- the whole library,
one cartridge, or one shelf -- and has the library's own reader model write, for each, a
question that passage answers and the key terms any right answer must contain. A term is
kept only if it appears in the passage, so the model cannot invent the answer key; the
passage's volume is the one a right answer should cite. The result is a plain JSON file
you can read, edit, and add to by hand.

Nothing is sent anywhere a question would not go: the sampled volumes are held to the
classification ceiling that applies to the model doing the writing (the remote ceiling,
when it is a provider's).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
RESULTS = HERE / "results"
GENERATED = RESULTS / "cases.json"
EXAMPLE = HERE / "cases.example.json"

UNANSWERABLE = [
    "What did the library's owner have for breakfast on the day the collection was started?",
]

_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "question": {"type": "string", "maxLength": 240},
        "key_terms": {"type": "array", "items": {"type": "string", "maxLength": 40}},
    },
    "required": ["answerable", "question", "key_terms"],
}

_PROMPT = """Below is a passage from "{title}" (section: {section}).

---
{text}
---

Write ONE question this passage answers, as someone would ask it of their library, and the
2 to 4 key terms that any correct answer to it must contain.

Rules:
- Ask about the substance, not about "the passage" or "the document".
- Specific enough that this passage is clearly the answer, not a generic question.
- Key terms: short (one to three words), copied exactly as they appear in the passage --
  names, numbers, commands, settings, technical terms. Not the question's own words.
- If the passage is boilerplate, a list of links, references, or has nothing to ask about,
  set answerable to false.

Return JSON with "answerable", "question" and "key_terms"."""


def resolve(path: Path | None) -> Path:
    """The cases to use: the path given, else this library's generated set, else the
    example (which only fits a library holding the papers it asks about)."""
    if path:
        return path
    return GENERATED if GENERATED.exists() else EXAMPLE


def load(path: Path) -> dict:
    return json.loads(path.read_text())


async def _allowed_levels(model: str) -> list[str] | None:
    from library_agent import classification
    from library_agent.llm import providers

    return classification.allowed_levels(None, remote=providers.is_remote(model))


async def _scope(db, cartridge: str | None, shelf: str | None) -> list | None:
    """Volume ids in the chosen scope, or None for the whole library."""
    from sqlalchemy import text

    if cartridge:
        ids = (
            (
                await db.execute(
                    text(
                        "select cd.document_id from cartridge_document cd join cartridge c"
                        " on c.id = cd.cartridge_id where lower(c.name) = lower(:n)"
                    ),
                    {"n": cartridge},
                )
            )
            .scalars()
            .all()
        )
        if not ids:
            raise SystemExit(f"no cartridge named {cartridge!r} on this library's rack")
        return list(ids)
    if shelf:
        from library_agent.library.shelving import expand_category_ids

        cats = (
            (
                await db.execute(
                    text("select id from category where lower(name) = lower(:n)"), {"n": shelf}
                )
            )
            .scalars()
            .all()
        )
        if not cats:
            raise SystemExit(f"no shelf named {shelf!r}")
        cats = await expand_category_ids(db, list(cats))
        ids = (
            (
                await db.execute(
                    text(
                        "select d.id from document d where d.shelf_id = any(:c) or exists ("
                        "select 1 from document_category dc where dc.document_id = d.id"
                        " and dc.category_id = any(:c))"
                    ),
                    {"c": cats},
                )
            )
            .scalars()
            .all()
        )
        return list(ids)
    return None


_PASSAGES = """
select c.id, c.document_id, d.title, coalesce(s.path, s.title, '') as section, c.text
from chunk c join document d on d.id = c.document_id
left join section s on s.id = c.section_id
where c.kind = 'text' and c.token_count >= 120 and not d.readings_only
  and (cast(:docs as uuid[]) is null or c.document_id = any(cast(:docs as uuid[])))
  and (cast(:levels as text[]) is null
       or coalesce(c.portion, d.classification, cast(:deflt as text)) = any(cast(:levels as text[])))
"""


def _found(term: str, text_: str) -> bool:
    t = " ".join(term.split()).lower()
    return len(t) >= 2 and t in " ".join(text_.split()).lower()


def _echoes(term: str, question: str) -> bool:
    words = set(re.findall(r"[a-z0-9_.-]+", question.lower()))
    parts = re.findall(r"[a-z0-9_.-]+", term.lower())
    return not parts or all(w in words for w in parts)


async def build(
    *, n: int = 10, cartridge: str | None = None, shelf: str | None = None, seed: int = 11
) -> dict:
    """Write a question set from this library. Returns the cases dict (also saved)."""
    from sqlalchemy import text

    from library_agent.classification import default_level
    from library_agent.config import settings
    from library_agent.db.session import session_scope
    from library_agent.llm.client import LLM

    model = settings().reader_model
    levels = await _allowed_levels(model)
    async with session_scope() as db:
        docs = await _scope(db, cartridge, shelf)
        rows = (
            await db.execute(
                text(_PASSAGES),
                {
                    "docs": [str(d) for d in docs] if docs is not None else None,
                    "levels": levels,
                    "deflt": default_level() if levels else None,
                },
            )
        ).all()
        # A thread's label makes a brief the library can write about from several volumes.
        thread = (
            await db.execute(
                text(
                    "select label from cluster where label is not null"
                    " and (cast(:docs as uuid[]) is null or exists (select 1 from cluster_claim cc"
                    "   where cc.cluster_id = cluster.id and cc.document_id = any(cast(:docs as uuid[]))))"
                    " order by document_count desc nulls last limit 1"
                ),
                {"docs": [str(d) for d in docs] if docs is not None else None},
            )
        ).scalar_one_or_none()
    if not rows:
        raise SystemExit("no passages to write questions from -- have volumes been added and read?")

    # One passage per volume in turn, so one long book cannot supply every question.
    rng = random.Random(seed)
    by_doc: dict = {}
    for r in rows:
        by_doc.setdefault(r.document_id, []).append(r)
    order = list(by_doc)
    rng.shuffle(order)
    for v in by_doc.values():
        rng.shuffle(v)

    questions: list[dict] = []
    used_docs: list[str] = []
    tried = 0
    async with LLM() as c:
        while len(questions) < n and tried < n * 4 and any(by_doc.values()):
            for d in order:
                if len(questions) >= n or tried >= n * 4:
                    break
                if not by_doc[d]:
                    continue
                r = by_doc[d].pop()
                tried += 1
                try:
                    out = await c.structured(
                        model,
                        _PROMPT.format(title=r.title, section=r.section or "—", text=r.text[:3000]),
                        _SCHEMA,
                        temperature=0.3,
                    )
                except Exception:  # noqa: BLE001, S112 -- one bad passage does not stop the set
                    continue
                q = " ".join(str(out.get("question") or "").split())
                # A term must be in the passage (the model cannot invent the key) and not
                # already in the question (an answer that echoes the question proves nothing).
                terms = [
                    t
                    for t in (out.get("key_terms") or [])
                    if _found(str(t), r.text) and not _echoes(str(t), q)
                ]
                if not out.get("answerable") or len(q) < 15 or not terms:
                    continue
                questions.append(
                    {
                        # Each group is one term; add synonyms to a group by hand if a
                        # right answer could phrase it another way.
                        "question": q,
                        "terms": [[t.lower()] for t in terms[:3]],
                        "volume": r.title,
                        "passage": str(r.id),
                    }
                )
                used_docs.append(str(r.document_id))
                print(f"  {len(questions)}/{n}  {q[:90]}", flush=True)

    cases: dict[str, Any] = {
        "_about": (
            "Generated by `./library bench init` from this library "
            f"({'cartridge ' + repr(cartridge) if cartridge else 'shelf ' + repr(shelf) if shelf else 'the whole library'}). "
            "Each question: groups of terms a right answer must contain (any one per group) and "
            "the volume it should cite. Edit freely: fix a question, add synonyms to a group, "
            "delete what does not fit."
        ),
        "scope": {"cartridge": cartridge, "shelf": shelf},
        # Exact terms from the passage: a right answer holds at least half of them.
        "terms_required": 0.5,
        "questions": questions,
        "unanswerable": UNANSWERABLE,
        "write_brief": (
            f"What does the library say about {thread}? Bring together what its volumes establish, and where they differ."
            if thread
            else f"Summarise what the library holds on {questions[0]['volume']} and how it relates to the rest."
        ),
        "sample_documents": sorted(set(used_docs)),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    return cases


async def sections(cases: dict, model: str, n: int = 3) -> list[dict]:
    """Sections for the section-reading test, from the volumes the questions came from
    (or any read volume), held to the ceiling that applies to `model`."""
    from sqlalchemy import text

    from library_agent.classification import default_level
    from library_agent.db.session import session_scope

    levels = await _allowed_levels(model)
    docs = cases.get("sample_documents") or None
    async with session_scope() as db:
        rows = (
            await db.execute(
                text(
                    "select d.title, coalesce(s.path, s.title, '') as path,"
                    " string_agg(c.text, E'\\n\\n' order by c.order_index) as body"
                    " from section s join document d on d.id = s.document_id"
                    " join chunk c on c.section_id = s.id and c.kind = 'text'"
                    " where not d.readings_only"
                    "  and (cast(:docs as uuid[]) is null or d.id = any(cast(:docs as uuid[])))"
                    "  and (cast(:levels as text[]) is null or coalesce(d.classification,"
                    "       cast(:deflt as text)) = any(cast(:levels as text[])))"
                    " group by d.title, s.path, s.title, s.id"
                    " having sum(length(c.text)) between 1500 and 6000"
                    "  and bool_and(c.portion is null or cast(:levels as text[]) is null"
                    "               or c.portion = any(cast(:levels as text[])))"
                    " order by md5(d.title || s.id::text) limit :n"
                ),
                {
                    "docs": docs,
                    "levels": levels,
                    "deflt": default_level() if levels else None,
                    "n": n,
                },
            )
        ).all()
    return [{"title": t, "path": p, "body": b} for t, p, b in rows]


def missing_volumes(cases: dict, titles: set[str]) -> list[str]:
    """Expected volumes this library does not hold -- a set written for another shelf."""
    want = {q["volume"] for q in cases.get("questions", [])}
    low = {t.lower() for t in titles}
    return sorted(w for w in want if not any(w.lower() in t for t in low))
