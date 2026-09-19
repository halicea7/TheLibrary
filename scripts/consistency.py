"""Answer consistency and correctness, end to end.

Retrieval has its own eval (recall@k over generated questions). This asks the other
question: when a person asks, do they get the right answer, and do they get it again?
Each question is asked N times without memory. For each run we record whether the
expected volume was cited, whether the answer contains the terms a correct answer must
contain, and the citation tally. A question the library cannot know is asked too, to
check that the librarian says so rather than inventing a source.

    uv run python scripts/consistency.py [--runs 3] [--model general|technical]
"""

import argparse, asyncio, json, statistics, sys, time
import httpx

API = "http://127.0.0.1:8077"

# (question, terms a correct answer must contain (any of each group), expected volume substring)
CASES = [
    ("What does RAPTOR build from the chunks of a document, and how?",
     [["tree"], ["cluster"], ["summar"]], "RAPTOR"),
    ("In Dense Passage Retrieval, what are the two encoders and what does each encode?",
     [["question", "query"], ["passage"], ["bert", "encoder"]], "Dense Passage Retrieval"),
    ("What does reciprocal rank fusion combine, and why does it not need score calibration?",
     [["rank"], ["score", "calibrat"]], "Retrieval Engineering"),
    ("Which two pre-training tasks does BERT use?",
     [["masked"], ["next sentence"]], "BERT"),
    ("What is HNSW and what does it trade off?",
     [["graph", "navigable"], ["neighbor", "neighbour"]], "navigable small world"),
]
UNKNOWABLE = "What did the library's owner have for breakfast on the day the collection was started?"


async def ask(c, q, model):
    t0 = time.time()
    r = await c.post("/api/v1/ask", json={"question": q, "remember": False, "model": model, "subjects": ["Machine Learning"]})
    r.raise_for_status()
    d = r.json()
    d["seconds"] = round(time.time() - t0, 1)
    return d


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default="general")
    a = ap.parse_args()
    rows = []
    async with httpx.AsyncClient(base_url=API, timeout=900) as c:
        for q, groups, expect in CASES:
            print(f"\n{q}")
            for i in range(a.runs):
                d = await ask(c, q, a.model)
                low = d["answer"].lower()
                terms_ok = all(any(t in low for t in g) for g in groups)
                cited_titles = [x["title"] for x in d["citations"]]
                vol_ok = any(expect.lower() in t.lower() for t in cited_titles)
                v = d["verified"]
                rows.append((q, i, terms_ok, vol_ok, v["emitted"], v["resolved"], d["seconds"]))
                print(f"  run {i+1}: terms {'ok ' if terms_ok else 'NO '} · cited {expect!r} {'ok ' if vol_ok else 'NO '} · "
                      f"{v['resolved']}/{v['emitted']} citations verified · {len(d['citations'])} sources · {d['seconds']}s")
        print(f"\n{UNKNOWABLE}")
        for i in range(a.runs):
            d = await ask(c, UNKNOWABLE, a.model)
            low = d["answer"].lower()
            honest = not d["citations"] and any(k in low for k in ("not", "no ", "cannot", "does not", "doesn't", "outside"))
            print(f"  run {i+1}: {'declines with no citation' if honest else 'SUSPECT'} · {len(d['citations'])} citations · {d['answer'][:90].replace(chr(10),' ')!r}")
            rows.append((UNKNOWABLE, i, honest, not d["citations"], d["verified"]["emitted"], d["verified"]["resolved"], d["seconds"]))

    n = len(rows)
    terms = sum(1 for r in rows if r[2]) / n
    vols = sum(1 for r in rows if r[3]) / n
    emitted = sum(r[4] for r in rows); resolved = sum(r[5] for r in rows)
    # consistency: for each question, did every run agree on terms and volume?
    per_q = {}
    for r in rows:
        per_q.setdefault(r[0], []).append((r[2], r[3]))
    consistent = sum(1 for v in per_q.values() if len({x for x in v}) == 1) / len(per_q)
    print(f"\n== {n} answers over {len(per_q)} questions × {a.runs} runs on '{a.model}' ==")
    print(f"correct terms present   {terms:.0%}")
    print(f"expected volume cited   {vols:.0%}")
    print(f"citations verified      {resolved}/{emitted} ({resolved/max(1,emitted):.0%})")
    print(f"runs agree per question {consistent:.0%}")
    print(f"median seconds/answer   {statistics.median(r[6] for r in rows)}")


asyncio.run(main())
