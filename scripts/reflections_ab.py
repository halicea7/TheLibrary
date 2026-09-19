"""Do Tier 2 reflections improve chat answers?

Retrieval said no (twice). This asks about synthesis: the same question, the same
passages, once with the library's margin notes attached beneath each passage and once
without, judged blind by the model on groundedness and usefulness, with the mechanical
citation-validity metric alongside as the sanity check that needs no judge.

Questions are generated from annotated volumes (Tier 2) so every question has notes
available to it. Retrieval is done once per question and shared between A and B, so
the only difference is the notes. The judge sees both answers in random order.

    uv run python scripts/reflections_ab.py [--n 20] [--model general]
"""

import argparse, asyncio, json, random, statistics, sys, uuid
from sqlalchemy import select, text

sys.path.insert(0, "src")
from library_agent.chat import answer as answer_mod  # noqa: E402
from library_agent.chat.citations import Source, build_sources, citation_validity, validate  # noqa: E402
from library_agent.config import settings  # noqa: E402
from library_agent.db.models import Artifact, ArtifactKind, Chunk, Document  # noqa: E402
from library_agent.db.session import session_scope  # noqa: E402
from library_agent.llm.ollama import Ollama  # noqa: E402
from library_agent.reading.prompts import SYSTEM_LIBRARIAN  # noqa: E402
from library_agent.retrieval.pipeline import CHAT_RETRIEVAL, retrieve  # noqa: E402

QGEN = {"type": "object", "properties": {"question": {"type": "string", "maxLength": 200}}, "required": ["question"]}
QGEN_PROMPT = """Here is a passage from a document titled "{title}", and the library's note on it.

Passage:
{text}

Note:
{note}

Write one question a careful reader would ask that this passage answers -- not a trivia
lookup, a question whose answer needs understanding what the passage means or implies.
Do not mention the passage or the document; ask as if to a colleague."""

JUDGE = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 600},
        "grounded_A": {"type": "integer", "minimum": 1, "maximum": 5},
        "grounded_B": {"type": "integer", "minimum": 1, "maximum": 5},
        "useful_A": {"type": "integer", "minimum": 1, "maximum": 5},
        "useful_B": {"type": "integer", "minimum": 1, "maximum": 5},
        "prefer": {"type": "string", "enum": ["A", "B", "tie"]},
    },
    "required": ["reasoning", "grounded_A", "grounded_B", "useful_A", "useful_B", "prefer"],
}
JUDGE_PROMPT = """You are judging two answers to the same question, written from the same passages.

Question: {q}

Passages the answers were allowed to use:
{ctx}

Answer A:
{a}

Answer B:
{b}

Score each 1-5 on grounded (every claim traceable to the passages; nothing invented) and
useful (answers what was asked, with the insight a careful reader would want, not a
restatement). Then say which you prefer overall, or tie. Reason first, briefly."""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="general")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    cfg = settings()
    model = cfg.chat_model_options.get(a.model, a.model)
    rnd = random.Random(a.seed)

    async with Ollama() as client, session_scope() as db:
        # Passages that have a note, from annotated volumes.
        rows = (await db.execute(text("""
            select c.id as chunk_id, c.text, d.title, a.text as note
            from artifact a join chunk c on c.id = a.target_id join document d on d.id = c.document_id
            where a.kind = 'reflection' and d.tier >= 2 and length(c.text) > 400
        """))).all()
        if not rows:
            print("no annotated passages"); return
        picks = rnd.sample(rows, min(a.n, len(rows)))
        print(f"{len(rows)} annotated passages; asking {len(picks)} questions on {model}\n")

        results = []
        for i, r in enumerate(picks, 1):
            q = (await client.structured(cfg.reader_model, QGEN_PROMPT.format(title=r.title, text=r.text[:1500], note=r.note[:500]), QGEN, system=SYSTEM_LIBRARIAN, temperature=0.4, seed=i)).get("question", "").strip()
            if not q:
                continue
            hits = await retrieve(db, q, config=CHAT_RETRIEVAL, client=client, limit=cfg.chat_passages)
            sources = build_sources(hits)
            # notes for the retrieved chunks
            notes = {}
            if hits:
                arts = (await db.execute(select(Artifact.target_id, Artifact.text).where(Artifact.kind == ArtifactKind.REFLECTION, Artifact.target_id.in_([h.chunk_id for h in hits])))).all()
                notes = {str(t): n for t, n in arts}
            async def answer(with_notes):
                msgs = answer_mod.build_messages(q, hits, sources, [], reflections=notes if with_notes else None)
                out = []
                async for kind, piece in client.chat_stream(model, msgs, temperature=0.3, seed=7):
                    if kind == "content": out.append(piece)
                raw = "".join(out); cleaned, used = validate(raw, sources)
                return cleaned, citation_validity(raw, sources)
            ans_a, m_a = await answer(False)
            ans_b, m_b = await answer(True)
            # blind: shuffle which is shown as A
            flip = rnd.random() < .5
            shown_a, shown_b = (ans_b, ans_a) if flip else (ans_a, ans_b)
            ctx = "\n\n".join(f"[{s.n}] {s.document_title}: {h.text[:600]}" for s, h in zip(sources, hits))
            j = await client.structured(cfg.reader_model, JUDGE_PROMPT.format(q=q, ctx=ctx, a=shown_a, b=shown_b), JUDGE, system="You are a careful, fair judge.", temperature=0.0, seed=3, think=True, num_predict=2500)
            # map back: 'A' shown == notes? if flip, shown A is the notes answer
            g_plain, g_notes = (j["grounded_B"], j["grounded_A"]) if flip else (j["grounded_A"], j["grounded_B"])
            u_plain, u_notes = (j["useful_B"], j["useful_A"]) if flip else (j["useful_A"], j["useful_B"])
            pref = j["prefer"]
            pref = "notes" if (pref == "A") == flip and pref != "tie" else ("plain" if pref != "tie" else "tie")
            results.append(dict(q=q, g_plain=g_plain, g_notes=g_notes, u_plain=u_plain, u_notes=u_notes, pref=pref, v_plain=m_a["validity"], v_notes=m_b["validity"], notes_available=len(notes)))
            print(f"{i:2d}. {'notes' if pref=='notes' else 'plain' if pref=='plain' else 'tie  '} · grounded {g_plain}/{g_notes} · useful {u_plain}/{u_notes} · validity {m_a['validity']:.2f}/{m_b['validity']:.2f} · {len(notes)} notes · {q[:70]}", flush=True)

    n = len(results)
    if not n:
        return
    print(f"\n== {n} questions, plain vs with notes ==")
    print(f"preferred: notes {sum(r['pref']=='notes' for r in results)} · plain {sum(r['pref']=='plain' for r in results)} · tie {sum(r['pref']=='tie' for r in results)}")
    print(f"grounded  mean: plain {statistics.mean(r['g_plain'] for r in results):.2f} · notes {statistics.mean(r['g_notes'] for r in results):.2f}")
    print(f"useful    mean: plain {statistics.mean(r['u_plain'] for r in results):.2f} · notes {statistics.mean(r['u_notes'] for r in results):.2f}")
    print(f"citation validity: plain {statistics.mean(r['v_plain'] for r in results):.3f} · notes {statistics.mean(r['v_notes'] for r in results):.3f}")
    print(f"questions where retrieved passages had notes: {sum(1 for r in results if r['notes_available'])}/{n}")
    json.dump(results, open("/tmp/reflections_ab.json", "w"), indent=1)


asyncio.run(main())
