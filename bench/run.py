"""Benchmarks for the models the library runs on: speed, reliability, correctness.

The retrieval eval (./library eval) asks whether the right passage comes back. This asks
the other question: given a model -- local, over the tunnel, or on a provider -- how fast
is it, how often does it break the library's contracts, and how often is it right?

Four suites, each run once per model:

    raw         the model alone, no library: a short reply's latency; a ~250-word stream's
                time to first token and output rate; four streams at once
    structured  the library's own structured calls -- Write's facet split, Tier 1's
                section reading, the conflict judge -- instrumented so a reply that fails
                to parse is counted, not silently retried. The judge runs on hand-made
                cases with known answers
    ask         end to end through the running library: questions with known answers,
                each asked --runs times; time to first word, whether the answer holds the
                terms it must, cites the volume it should, verifies every citation, and
                agrees with itself -- plus a question the library cannot answer
    write       one short composition: step timings, citations verified, claims flagged

    ./library bench init                          # write questions from YOUR library, once
    ./library bench                               # the chat model, every suite
    ./library bench --models qwen3:30b-a3b acme:some-model --runs 3
    ./library bench --suites raw,structured --models acme:new-model

Everything a run measures lands in bench/results/<stamp>-<label>/: raw.jsonl (one line per
measurement), summary.json, report.md. That directory is gitignored: results name your
providers and quote your library. The library must be running for `ask` and `write`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cases as cases_mod  # bench/cases.py
import httpx

from library_agent.llm import providers
from library_agent.llm.client import LLM
from library_agent.llm.ollama import OllamaError

HERE = Path(__file__).parent
RESULTS = HERE / "results"
API = "http://127.0.0.1:8077"

# ----------------------------------------------------------------------------- recording


class Recorder:
    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []

    def add(self, **row: Any) -> dict:
        row["at"] = round(time.time(), 2)
        self.rows.append(row)
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return row

    def where(self, **kw: Any) -> list[dict]:
        return [r for r in self.rows if all(r.get(k) == v for k, v in kw.items())]


def pct(xs: list[float], p: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 2)


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 2) if xs else None


def rate(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "—"


def say(*a: Any) -> None:
    print(*a, flush=True)


# ----------------------------------------------------------------------------- raw


RAW_PROMPT = (
    "Explain, in about 200 words of plain prose, how a hash table handles collisions. "
    "No headings, no lists."
)


async def _stream_once(c: LLM, model: str) -> dict:
    t0 = time.time()
    first_any = first_content = None
    thinking = content = ""
    async for kind, piece in c.chat_stream(
        model, [{"role": "user", "content": RAW_PROMPT}], temperature=0.3
    ):
        now = time.time() - t0
        first_any = first_any if first_any is not None else now
        if kind == "thinking":
            thinking += piece
        else:
            first_content = first_content if first_content is not None else now
            content += piece
    total = time.time() - t0
    gen = total - (first_content or total)
    # Tokens estimated at four characters each: the same yardstick for every backend,
    # since not every server reports usage on a stream.
    return {
        "ttft": round(first_any or total, 2),
        "first_word": round(first_content, 2) if first_content is not None else None,
        "total": round(total, 2),
        "thinking_chars": len(thinking),
        "content_chars": len(content),
        "tok_per_s": round(len(content) / 4 / gen, 1) if gen > 0.2 else None,
    }


async def suite_raw(rec: Recorder, model: str, runs: int) -> None:
    async with LLM() as c:
        for i in range(max(runs, 3)):
            t0 = time.time()
            try:
                await c.generate(model, "Reply with the single word: ready", num_predict=20)
                rec.add(
                    suite="raw",
                    test="short",
                    model=model,
                    ok=True,
                    seconds=round(time.time() - t0, 2),
                )
            except Exception as exc:  # noqa: BLE001
                rec.add(suite="raw", test="short", model=model, ok=False, error=str(exc)[:200])
        for i in range(runs):
            try:
                rec.add(
                    suite="raw", test="stream", model=model, ok=True, **await _stream_once(c, model)
                )
            except Exception as exc:  # noqa: BLE001
                rec.add(suite="raw", test="stream", model=model, ok=False, error=str(exc)[:200])
        t0 = time.time()
        outs = await asyncio.gather(
            *(_stream_once(c, model) for _ in range(4)), return_exceptions=True
        )
        wall = time.time() - t0
        good = [o for o in outs if isinstance(o, dict)]
        rec.add(
            suite="raw",
            test="concurrent4",
            model=model,
            ok=len(good) == 4,
            errors=[str(o)[:120] for o in outs if not isinstance(o, dict)],
            wall=round(wall, 2),
            agg_tok_per_s=round(sum(o["content_chars"] for o in good) / 4 / wall, 1)
            if good
            else None,
            ttft_p50=pct([o["ttft"] for o in good], 0.5),
        )
    say("  raw: done")


# ----------------------------------------------------------------------------- structured


class CountingLLM(LLM):
    """Every structured call made one attempt at a time, so a reply that fails to parse
    is recorded rather than quietly retried. Three attempts, as the library allows."""

    def __init__(self, rec: Recorder, model: str):
        super().__init__()
        self.rec, self.model, self.task = rec, model, ""

    async def structured(self, model: str, prompt: str, schema: dict, **kw: Any) -> dict:
        kw.pop("retries", None)
        reasons: list[str] = []
        t0 = time.time()
        for attempt in range(1, 4):
            try:
                out = await super().structured(model, prompt, schema, retries=0, **kw)
                self.rec.add(
                    suite="structured",
                    task=self.task,
                    model=self.model,
                    ok=True,
                    attempts=attempt,
                    first_try=attempt == 1,
                    reasons=reasons,
                    seconds=round(time.time() - t0, 2),
                )
                return out
            except OllamaError as exc:
                m = re.search(
                    r"(malformed JSON|placeholder values|echoed the prompt|HTTP \d+)", str(exc)
                )
                reasons.append(m.group(1) if m else str(exc)[:80])
        self.rec.add(
            suite="structured",
            task=self.task,
            model=self.model,
            ok=False,
            attempts=3,
            first_try=False,
            reasons=reasons,
            seconds=round(time.time() - t0, 2),
        )
        raise OllamaError("unusable after 3 attempts: " + "; ".join(reasons))


BRIEFS = [
    "Compare how BERT-style cross-encoders and ColBERT's late interaction trade accuracy for speed.",
    "What are the failure modes of reciprocal rank fusion, and when does a learned reranker beat it?",
    "Explain how HNSW builds its graph, and what its parameters trade off in recall, memory and build time.",
]

# The judge's cases, written for this benchmark. Two disagree; two look alike but do not
# (two different clusters' facts; two compatible properties of one function).
JUDGE_CASES = [
    {
        "label": "Raft commit rule",
        "expect": True,
        "claims": [
            "In Raft, a leader commits a log entry once a majority of the cluster has stored it.",
            "In Raft, an entry is committed as soon as any single follower acknowledges it.",
        ],
        "titles": ["Consensus Notes A", "Consensus Notes B"],
    },
    {
        "label": "TLS 1.3 key exchange",
        "expect": True,
        "claims": [
            "TLS 1.3 removed static RSA key exchange; every handshake uses ephemeral Diffie-Hellman.",
            "TLS 1.3 still allows static RSA key exchange for servers that need it.",
        ],
        "titles": ["Protocol Guide", "Server Hardening Notes"],
    },
    {
        "label": "Cluster head nodes",
        "expect": False,
        "claims": [
            "The head node of the Pegasus cluster is login01.",
            "The head node of the Orion cluster is hydra01.",
        ],
        "titles": ["Pegasus User Guide", "Orion User Guide"],
        "scopes": {
            "Pegasus User Guide": "How to use the Pegasus cluster.",
            "Orion User Guide": "How to use the Orion cluster.",
        },
    },
    {
        "label": "BM25 properties",
        "expect": False,
        "claims": [
            "BM25 saturates term frequency, so repeating a word has diminishing returns.",
            "BM25 normalises for document length, so long documents are not favoured.",
        ],
        "titles": ["Ranking Functions", "IR Lecture Notes"],
    },
]


async def suite_structured(rec: Recorder, model: str, runs: int, cases: dict) -> None:
    from library_agent.chat import compose
    from library_agent.library.contradictions import judge_cluster
    from library_agent.reading import prompts
    from library_agent.reading.genre import CLAIMS_BY_GENRE

    c = CountingLLM(rec, model)
    try:
        c.task = "facets"
        for b in BRIEFS:
            for _ in range(runs):
                await compose._facets(c, model, b)  # falls back on failure; recorded above
        c.task = "section_reading"
        for s in await cases_mod.sections(cases, model):
            for _ in range(runs):
                try:
                    out = await c.structured(
                        model,
                        prompts.SECTION_PROMPT.format(
                            title=s["title"],
                            orientation="",
                            previous="",
                            section_path=s["path"],
                            text=s["body"],
                            claims_guidance=CLAIMS_BY_GENRE["paper"],
                            category_guidance=prompts.category_guidance([]),
                        ),
                        prompts.SECTION_SCHEMA,
                        system=prompts.SYSTEM_LIBRARIAN,
                    )
                    rec.add(
                        suite="structured",
                        task="section_quality",
                        model=model,
                        claims=len(out.get("claims") or []),
                        summary_words=len(str(out.get("summary") or "").split()),
                    )
                except OllamaError:
                    pass
        c.task = "conflict_judge"
        for case in JUDGE_CASES:
            for _ in range(runs):
                t0 = time.time()
                v = await judge_cluster(
                    c,
                    model,
                    case["label"],
                    case["claims"],
                    case["titles"],
                    claim_sources=case["titles"],
                    scopes=case.get("scopes"),
                )
                got = bool(v and v.get("disagreement"))
                rec.add(
                    suite="structured",
                    task="judge_verdict",
                    model=model,
                    case=case["label"],
                    ok=v is not None,
                    correct=got == case["expect"],
                    expected=case["expect"],
                    got=got,
                    votes=(v or {}).get("votes"),
                    seconds=round(time.time() - t0, 2),
                )
    finally:
        await c.aclose()
    say("  structured: done")


# ----------------------------------------------------------------------------- ask


async def _ask(client: httpx.AsyncClient, model: str, q: str, effort: str) -> dict:
    t0 = time.time()
    ev = None
    out: dict[str, Any] = {"thinking_at": None, "first_word": None, "sources": [], "error": None}
    conv = None
    async with client.stream(
        "POST",
        "/api/chat",
        json={"message": q, "conversational": False, "model": model, "effort": effort},
    ) as r:
        async for line in r.aiter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            now = round(time.time() - t0, 2)
            if ev == "conversation":
                conv = json.loads(raw)["conversation_id"]
            elif ev == "thinking" and out["thinking_at"] is None:
                out["thinking_at"] = now
            elif ev == "token" and out["first_word"] is None:
                out["first_word"] = now
            elif ev == "sources":
                out["sources"] = json.loads(raw)
            elif ev == "withheld":
                out["withheld"] = json.loads(raw)["count"]
            elif ev == "error":
                out["error"] = raw[:300]
            elif ev == "done":
                d = json.loads(raw)
                out.update(
                    answer=d["answer"],
                    cited=d.get("cited", []),
                    emitted=d.get("markers_emitted", 0),
                    resolved=d.get("markers_resolved", 0),
                    flags=len(d.get("flags") or []),
                    ceiling=(d.get("classification") or {}).get("ceiling"),
                )
    out["total"] = round(time.time() - t0, 2)
    if conv:  # a benchmark's questions are not the person's history
        try:
            await client.delete(f"/api/conversations/{conv}")
        except httpx.HTTPError:
            pass
    return out


DECLINES = (
    "does not contain",
    "doesn't contain",
    "no information",
    "not contain",
    "cannot answer",
    "can't answer",
    "does not address",
    "outside",
    "not in the library",
    "no passage",
    "not covered",
    "none of the",
)


async def suite_ask(rec: Recorder, model: str, runs: int, cases: dict, effort: str) -> None:
    # A hand-written set lists what a right answer must say (all of it); a generated set
    # lists exact terms from the passage, which a right answer may phrase otherwise.
    share = float(cases.get("terms_required", 1.0))
    async with httpx.AsyncClient(base_url=API, timeout=900) as client:
        for case in cases["questions"]:
            for i in range(runs):
                d = await _ask(client, model, case["question"], effort)
                low = (d.get("answer") or "").lower()
                flat = " ".join(low.split())
                titles = [s["title"] for s in d["sources"] if s["n"] in d.get("cited", [])]
                rec.add(
                    suite="ask",
                    model=model,
                    question=case["question"],
                    run=i,
                    ok=d["error"] is None,
                    error=d["error"],
                    first_word=d["first_word"],
                    thinking_at=d["thinking_at"],
                    total=d["total"],
                    words=len(low.split()),
                    terms_found=(found := sum(any(t in flat for t in g) for g in case["terms"])),
                    terms_total=len(case["terms"]),
                    terms_ok=found >= math.ceil(share * len(case["terms"])),
                    volume_ok=any(case["volume"].lower() in t.lower() for t in titles),
                    emitted=d.get("emitted", 0),
                    resolved=d.get("resolved", 0),
                    sources=len(d["sources"]),
                    flags=d.get("flags"),
                    ceiling=d.get("ceiling"),
                    withheld=d.get("withheld", 0),
                )
        for q in cases.get("unanswerable", []):
            for i in range(runs):
                d = await _ask(client, model, q, effort)
                low = (d.get("answer") or "").lower()
                rec.add(
                    suite="ask",
                    model=model,
                    question=q,
                    run=i,
                    unanswerable=True,
                    ok=d["error"] is None,
                    declined=any(k in low for k in DECLINES),
                    first_word=d["first_word"],
                    total=d["total"],
                    emitted=d.get("emitted", 0),
                    resolved=d.get("resolved", 0),
                )
    say("  ask: done")


# ----------------------------------------------------------------------------- write


async def suite_write(rec: Recorder, model: str, brief: str) -> None:
    from library_agent.chat.compose import compositions_dir, slugify

    t0 = time.time()
    ev = None
    row: dict[str, Any] = {"suite": "write", "model": model, "ok": False, "sections": []}
    async with (
        httpx.AsyncClient(base_url=API, timeout=3600) as client,
        client.stream(
            "POST", "/api/compose", json={"brief": brief, "model": model, "length": "short"}
        ) as r,
    ):
        async for line in r.aiter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if ev == "coverage":
                row["coverage"] = json.loads(raw).get("verdict")
            elif ev == "section_done":
                d = json.loads(raw)
                row["sections"].append({"cited": len(d["cited"]), "flags": len(d["flags"])})
            elif ev == "error":
                row["error"] = raw[:300]
            elif ev == "done":
                d = json.loads(raw)
                row.update(
                    ok=True,
                    emitted=d["markers_emitted"],
                    resolved=d["markers_resolved"],
                    flags=d["flags"],
                    sources=d["sources"],
                    timings=d.get("timings"),
                )
                # The library saved it as a composition; a benchmark's is not one.
                tag = d["id"][:8]
                for f in compositions_dir().glob(f"{slugify(d['title'])}-{tag}.md"):
                    f.unlink(missing_ok=True)
    row["total"] = round(time.time() - t0, 1)
    rec.add(**row)
    say("  write: done")


# ----------------------------------------------------------------------------- report


def report(rec: Recorder, models: list[str], meta: dict) -> str:
    L: list[str] = [f"# Model benchmark — {meta['label']}", ""]
    L += [
        (
            f"*{meta['when']} · library commit `{meta['commit']}` · {meta['volumes']} volumes · "
            f"runs per case: {meta['runs']} · effort: {meta['effort']}*"
        ),
        "",
    ]

    def table(title: str, rows: list[tuple[str, list[Any]]], note: str = "") -> None:
        L.extend([f"### {title}", ""])
        if note:
            L.extend([note, ""])
        L.append("| | " + " | ".join(f"`{m}`" for m in models) + " |")
        L.append("|---|" + "---|" * len(models))
        for name, vals in rows:
            L.append(f"| {name} | " + " | ".join("—" if v is None else str(v) for v in vals) + " |")
        L.append("")

    def per(fn):
        return [fn(m) for m in models]

    if rec.where(suite="raw"):
        L.append("## Speed (the model alone)")
        L.append("")
        s = lambda m, t: rec.where(suite="raw", model=m, test=t)
        table(
            "Short reply",
            [
                (
                    "p50 seconds",
                    per(lambda m: pct([r["seconds"] for r in s(m, "short") if r["ok"]], 0.5)),
                ),
                (
                    "p95 seconds",
                    per(lambda m: pct([r["seconds"] for r in s(m, "short") if r["ok"]], 0.95)),
                ),
                (
                    "succeeded",
                    per(lambda m: rate(sum(r["ok"] for r in s(m, "short")), len(s(m, "short")))),
                ),
            ],
        )
        table(
            "A ~250-word answer, streamed",
            [
                (
                    "time to first token, p50 (s)",
                    per(lambda m: pct([r["ttft"] for r in s(m, "stream") if r["ok"]], 0.5)),
                ),
                (
                    "time to first answer word, p50 (s)",
                    per(lambda m: pct([r["first_word"] for r in s(m, "stream") if r["ok"]], 0.5)),
                ),
                (
                    "total, p50 (s)",
                    per(lambda m: pct([r["total"] for r in s(m, "stream") if r["ok"]], 0.5)),
                ),
                (
                    "output tokens/s, mean",
                    per(lambda m: mean([r["tok_per_s"] for r in s(m, "stream") if r["ok"]])),
                ),
                (
                    "reasoning before answering (chars, mean)",
                    per(lambda m: mean([r["thinking_chars"] for r in s(m, "stream") if r["ok"]])),
                ),
            ],
            "Tokens are estimated at four characters, the same for every backend.",
        )
        table(
            "Four streams at once",
            [
                (
                    "all four completed",
                    per(lambda m: "yes" if all(r["ok"] for r in s(m, "concurrent4")) else "no"),
                ),
                ("wall seconds", per(lambda m: (s(m, "concurrent4") or [{}])[0].get("wall"))),
                (
                    "aggregate tokens/s",
                    per(lambda m: (s(m, "concurrent4") or [{}])[0].get("agg_tok_per_s")),
                ),
            ],
        )

    if rec.where(suite="structured"):
        L.append("## Reliability (the library's structured calls)")
        L.append("")
        for task, name in (
            ("facets", "Write: splitting a brief into parts"),
            ("section_reading", "Tier 1: reading a section"),
            ("conflict_judge", "The conflict judge (each vote)"),
        ):
            rs = lambda m, task=task: rec.where(suite="structured", model=m, task=task)
            table(
                name,
                [
                    (
                        "valid on the first try",
                        per(lambda m, rs=rs: rate(sum(r["first_try"] for r in rs(m)), len(rs(m)))),
                    ),
                    (
                        "valid within 3 tries",
                        per(lambda m, rs=rs: rate(sum(r["ok"] for r in rs(m)), len(rs(m)))),
                    ),
                    (
                        "failure reasons",
                        per(
                            lambda m, rs=rs: (
                                ", ".join(sorted({x for r in rs(m) for x in r["reasons"]}))
                                or "none"
                            )
                        ),
                    ),
                    (
                        "seconds per call, p50",
                        per(lambda m, rs=rs: pct([r["seconds"] for r in rs(m)], 0.5)),
                    ),
                ]
                + (
                    [
                        (
                            "claims per section, mean",
                            per(
                                lambda m: mean(
                                    [
                                        r["claims"]
                                        for r in rec.where(
                                            suite="structured", model=m, task="section_quality"
                                        )
                                    ]
                                )
                            ),
                        ),
                        (
                            "summary words, mean",
                            per(
                                lambda m: mean(
                                    [
                                        r["summary_words"]
                                        for r in rec.where(
                                            suite="structured", model=m, task="section_quality"
                                        )
                                    ]
                                )
                            ),
                        ),
                    ]
                    if task == "section_reading"
                    else []
                ),
            )
        jv = lambda m: rec.where(suite="structured", model=m, task="judge_verdict")
        table(
            "Correctness: the conflict judge on known cases",
            [
                (
                    "right verdict",
                    per(lambda m: rate(sum(r["correct"] for r in jv(m)), len(jv(m)))),
                ),
                (
                    "real disagreements found",
                    per(
                        lambda m: rate(
                            sum(r["correct"] for r in jv(m) if r["expected"]),
                            sum(1 for r in jv(m) if r["expected"]),
                        )
                    ),
                ),
                (
                    "look-alikes left alone",
                    per(
                        lambda m: rate(
                            sum(r["correct"] for r in jv(m) if not r["expected"]),
                            sum(1 for r in jv(m) if not r["expected"]),
                        )
                    ),
                ),
                (
                    "wrong on",
                    per(
                        lambda m: (
                            ", ".join(sorted({r["case"] for r in jv(m) if not r["correct"]}))
                            or "none"
                        )
                    ),
                ),
            ],
            "Two pairs genuinely disagree; two look alike but don't (two clusters' head nodes; two compatible properties of BM25).",
        )

    if rec.where(suite="ask"):
        L.append("## Correctness (Ask, end to end)")
        L.append("")
        a = lambda m: [r for r in rec.where(suite="ask", model=m) if not r.get("unanswerable")]
        u = lambda m: [r for r in rec.where(suite="ask", model=m) if r.get("unanswerable")]

        def agree(m: str) -> str:
            by: dict[str, set] = {}
            for r in a(m):
                by.setdefault(r["question"], set()).add((r["terms_ok"], r["volume_ok"]))
            return rate(sum(len(v) == 1 for v in by.values()), len(by))

        table(
            "Answers",
            [
                ("completed", per(lambda m: rate(sum(r["ok"] for r in a(m)), len(a(m))))),
                (
                    "holds the key terms (all, or the set's share)",
                    per(lambda m: rate(sum(r["terms_ok"] for r in a(m)), len(a(m)))),
                ),
                (
                    "key terms present, mean share",
                    per(
                        lambda m: (
                            f"{100 * sum(r['terms_found'] for r in a(m)) / max(1, sum(r['terms_total'] for r in a(m))):.0f}%"
                            if a(m)
                            else None
                        )
                    ),
                ),
                (
                    "cites the volume it should",
                    per(lambda m: rate(sum(r["volume_ok"] for r in a(m)), len(a(m)))),
                ),
                (
                    "citations verified",
                    per(
                        lambda m: rate(
                            sum(r["resolved"] for r in a(m)), sum(r["emitted"] for r in a(m))
                        )
                    ),
                ),
                ("runs agree with each other", per(agree)),
                (
                    "declines what it cannot know",
                    per(lambda m: rate(sum(r["declined"] for r in u(m)), len(u(m)))),
                ),
                (
                    "claims flagged by review (deep only)",
                    per(lambda m: sum(r.get("flags") or 0 for r in a(m))),
                ),
            ],
        )
        table(
            "Answer speed",
            [
                ("first word, p50 (s)", per(lambda m: pct([r["first_word"] for r in a(m)], 0.5))),
                ("first word, p95 (s)", per(lambda m: pct([r["first_word"] for r in a(m)], 0.95))),
                ("whole answer, p50 (s)", per(lambda m: pct([r["total"] for r in a(m)], 0.5))),
                ("whole answer, p95 (s)", per(lambda m: pct([r["total"] for r in a(m)], 0.95))),
                ("words per answer, mean", per(lambda m: mean([r["words"] for r in a(m)]))),
                (
                    "classification ceiling applied",
                    per(lambda m: ", ".join(sorted({str(r.get("ceiling")) for r in a(m)}))),
                ),
            ],
            "Includes retrieval and reranking, which are the same for every model.",
        )

    if rec.where(suite="write"):
        L.append("## Write (one short composition)")
        L.append("")
        w = lambda m: (rec.where(suite="write", model=m) or [{}])[0]
        steps = sorted({k for m in models for k in (w(m).get("timings") or {}) if k != "elapsed"})
        table(
            "Result",
            [
                (
                    "completed",
                    per(lambda m: "yes" if w(m).get("ok") else f"no: {w(m).get('error', '')[:60]}"),
                ),
                ("coverage verdict", per(lambda m: w(m).get("coverage"))),
                ("sections", per(lambda m: len(w(m).get("sections") or []))),
                (
                    "citations verified",
                    per(lambda m: rate(w(m).get("resolved", 0), w(m).get("emitted", 0))),
                ),
                ("claims flagged by review", per(lambda m: w(m).get("flags"))),
                ("total seconds", per(lambda m: w(m).get("total"))),
            ],
        )
        table(
            "Where the time went (seconds)",
            [(k, per(lambda m, k=k: (w(m).get("timings") or {}).get(k))) for k in steps],
        )
    return "\n".join(L)


# ----------------------------------------------------------------------------- main


async def _busy() -> str | None:
    """Anything on the shared GPU that a benchmark would slow, or be slowed by."""
    try:
        async with httpx.AsyncClient(base_url=API, timeout=10) as c:
            h = (await c.get("/api/health")).json()
    except httpx.HTTPError:
        return "the library is not running (ask and write need it)"
    g = h.get("generations") or {}
    if g.get("in_flight"):
        return f"{g['in_flight']} generation(s) in flight -- a composition may be running"
    return None


async def _missing(cases: dict) -> list[str]:
    from sqlalchemy import text

    from library_agent.db.session import session_scope

    async with session_scope() as db:
        titles = set((await db.execute(text("select title from document"))).scalars())
    return cases_mod.missing_volumes(cases, titles)


async def init_main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="./library bench init",
        description="Write the benchmark's questions from this library (bench/results/cases.json).",
    )
    ap.add_argument("--n", type=int, default=10, help="questions to write (default 10)")
    ap.add_argument("--cartridge", help="only from this cartridge's volumes")
    ap.add_argument("--shelf", help="only from this shelf (and its sub-shelves)")
    a = ap.parse_args(argv)
    say(f"writing {a.n} questions from this library with its reader model...")
    cases = await cases_mod.build(n=a.n, cartridge=a.cartridge, shelf=a.shelf)
    say(
        f"\n{len(cases['questions'])} questions -> {cases_mod.GENERATED}\n"
        "Read them over and edit freely, then: ./library bench"
    )


async def main() -> None:
    if sys.argv[1:2] == ["init"]:
        await init_main(sys.argv[2:])
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--models",
        nargs="+",
        help="model names; a provider's as id:model (default: the chat model)",
    )
    ap.add_argument("--suites", default="raw,structured,ask,write")
    ap.add_argument("--runs", type=int, default=2, help="runs per case (default 2)")
    ap.add_argument("--effort", default="normal", help="Ask's effort: quick, normal or deep")
    ap.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="question set (default: bench/results/cases.json from `bench init`)",
    )
    ap.add_argument("--label", default="", help="a name for this run's directory")
    ap.add_argument("--force", action="store_true", help="run even if the GPU looks busy")
    a = ap.parse_args()
    suites = [s.strip() for s in a.suites.split(",") if s.strip()]
    models = a.models or [providers.model_for("chat_general")]
    cases_path = cases_mod.resolve(a.cases)
    cases = cases_mod.load(cases_path)
    if "ask" in suites or "write" in suites:
        missing = await _missing(cases)
        if missing and len(missing) == len({q["volume"] for q in cases["questions"]}):
            sys.exit(
                f"{cases_path.name} asks about volumes this library doesn't hold "
                f"({', '.join(missing[:3])}...). Write a set from yours first:\n"
                "    ./library bench init            # or: --cartridge NAME / --shelf NAME"
            )
        if missing:
            say(
                f"note: {len(missing)} expected volume(s) not on this shelf: {', '.join(missing[:3])}"
            )

    why = await _busy()
    if (
        why
        and not a.force
        and (
            ("ask" in suites or "write" in suites)
            or any(not providers.is_remote(m) for m in models)
        )
    ):
        sys.exit(f"not now: {why}. Use --force to run anyway.")

    stamp = datetime.now().astimezone().strftime("%Y-%m-%d-%H%M")
    label = re.sub(
        r"[^a-z0-9-]+", "-", (a.label or "-".join(m.split(":")[-1] for m in models)).lower()
    )[:60].strip("-")
    out = RESULTS / f"{stamp}-{label}"
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(out / "raw.jsonl")
    try:
        commit = subprocess.run(  # noqa: ASYNC221 -- once, before anything is measured
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=HERE,
            check=False,
        ).stdout.strip()
    except OSError:
        commit = "?"
    try:
        async with httpx.AsyncClient(base_url=API, timeout=10) as c:
            volumes = (await c.get("/api/health")).json().get("documents")
    except httpx.HTTPError:
        volumes = "?"
    meta = {
        "label": label,
        "when": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
        "commit": commit,
        "volumes": volumes,
        "runs": a.runs,
        "effort": a.effort,
        "models": models,
        "suites": suites,
        "cases": str(cases_path),
    }
    say(f"benchmark → {out}")
    started = time.time()
    for m in models:
        say(f"\n{m}")
        for s in suites:
            try:
                if s == "raw":
                    await suite_raw(rec, m, a.runs)
                elif s == "structured":
                    await suite_structured(rec, m, a.runs, cases)
                elif s == "ask":
                    await suite_ask(rec, m, a.runs, cases, a.effort)
                elif s == "write":
                    await suite_write(rec, m, cases["write_brief"])
            except Exception as exc:  # noqa: BLE001 -- one suite failing is itself a finding
                rec.add(
                    suite=s,
                    model=m,
                    ok=False,
                    crashed=True,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                )
                say(f"  {s}: crashed -- {exc}")
    meta["minutes"] = round((time.time() - started) / 60, 1)
    md = report(rec, models, meta)
    (out / "report.md").write_text(md + f"\n\n*Ran in {meta['minutes']} minutes.*\n")
    (out / "summary.json").write_text(json.dumps(meta, indent=1))
    say(f"\nreport: {out / 'report.md'}")


if __name__ == "__main__":
    asyncio.run(main())
