"""The audit layer: what separates a source's claim from a fact, shared by Ask and Write.

Retrieval breadth and review strictness are separate dials, and they used to live on
different sides: Ask's deep effort searched widely and said whatever the passages said,
while Write searched narrowly and reviewed every section. A paper's case for its own design
("understandable", "more predictable", its own benchmark) came through deep Ask as settled
fact. Both now share the rules below -- in the writing prompt, so they shape the prose, and
in the review, so a sentence that breaks them is flagged against the passages it cites.

Review stays flag-only: the prose is left as written and the reader weighs the flags."""

from __future__ import annotations

import logging
import re
from typing import Any

from library_agent.llm.ollama import is_placeholder

log = logging.getLogger(__name__)

# Handed to the writer (Ask and Write both). Terse, because it sits beside long context.
EVIDENCE_RULES = """Keep a source's claims apart from facts:
- A work's claims about ITS OWN design, method or product — that it is simpler, more
  understandable, faster, safer, "more predictable" — are its authors' claims. Attribute them
  ("the Raft authors argue [n]", "the vendor states [n]"); never restate them as settled fact.
- A measurement holds under its conditions. Keep the conditions (setup, cluster size,
  timeouts, hardware, workload) with the number; do not generalise one benchmark.
- Judgements such as "opaque", "elegant", "notoriously hard" are opinions: attribute them.
- Do not chain passages into a mechanism none of them states ("complex, so implementations
  err, so tail latency rises"). If that link is your inference, say it is your inference and
  leave it uncited.
- Before saying the library lacks something, check the passages you were given; if one
  covers it, use it instead."""

REVIEW_SYSTEM = """You are a careful reviewer of writing produced from a research library, and
you are hard on it. You are given the question or brief, the text, and the numbered passages
it was written from. Find only claims that reach past the evidence or miss the question — do
not rewrite, do not praise, do not restate. A claim is a flaw when:
- the passage it cites supports part of the statement but not the whole of it;
- it is absolute (never, always, cannot, guarantees, impossible, no way) where the sources
  are conditional, or hold only under assumptions the sources state;
- it generalises from a single case or benchmark, or drops the conditions a measurement was
  taken under, or asserts a cause the sources only correlate;
- it presents a source's claim about its own design, method or product (simpler, faster,
  more understandable, more predictable) as an established fact instead of attributing it
  to the authors or vendor;
- it presents a subjective judgement from a source as objective fact;
- it chains passages into a mechanism none of them states, presented as cited fact;
- it says the library does not cover something that one of the given passages does cover;
- it is presented as established but no passage actually supports it;
- it treats an intended rejection as a failure — an access correctly denied, a login that
  failed by design, is the security control WORKING, not a vulnerability;
- its mechanism does not bear on the question — an example pulled in by a shared word, not
  by the mechanism asked about, or a neighbouring subject substituted for the one asked;
- it merges two related protocols, tools, modes or identifiers that differ in mechanism;
- it asserts prevalence, defaults or motivation ("widespread", "rarely changed", "vendors
  cut corners") that no passage supports.
A well-supported, on-question text has no flags. Do not invent problems to have something
to say. Write each flag in full — never leave a sentence unfinished."""

REVIEW_PROMPT = """The question or brief: {brief}

Text under review{heading}:

{body}

Passages it was written from:
{context}

Return `flags`: each a claim that reaches past its evidence or misses the question, quoting
the exact sentence from the text as `quote`, saying in `issue` what is wrong (the evidence does
not carry it; it adopts a source's claim about itself as fact; it answers a different
question; it calls a control working a failure), and in `condition` the circumstance under
which the claim is false or beside the point. Empty if the text is sound and on-question."""

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "flags": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string", "maxLength": 300},
                    "issue": {"type": "string", "maxLength": 550},
                    "condition": {"type": "string", "maxLength": 550},
                },
                "required": ["quote", "issue"],
            },
        }
    },
    "required": ["flags"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def anchored(quote: str, body: str) -> bool:
    """A flag's quote must actually be in the text -- so a placeholder ("…") or a
    hallucinated sentence is dropped rather than shown beside real prose."""
    q, b = _norm(quote), _norm(body)
    words = q.split()
    if len(words) < 3:
        return False
    if q in b or " ".join(words[:6]) in b:
        return True
    qset = set(words)
    return len(qset & set(b.split())) / len(qset) >= 0.75


def clip(s: str, n: int) -> str:
    """Trim to a length, but at a sentence or word boundary so a flag never ends
    mid-word."""
    s = " ".join((s or "").split())
    if len(s) <= n:
        return s
    cut = s[:n]
    for sep in (". ", "; ", ", ", " "):
        i = cut.rfind(sep)
        if i > n // 2:
            return cut[: i + (1 if sep == " " else len(sep))].rstrip() + "…"
    return cut.rstrip() + "…"


async def review(
    client, model, body: str, context: str, *, brief: str = "", heading: str = ""
) -> list[dict]:
    """Read finished text against its passages and flag where it reaches past them or
    misses the question. Flag-only: it never touches the prose. A flag whose quote is not
    in the text is dropped -- which also keeps out the "…" placeholders."""
    if len(body.strip()) < 120 or not context.strip():
        return []
    try:
        out = await client.structured(
            model,
            REVIEW_PROMPT.format(
                brief=brief or "(none given)",
                heading=f' (the section "{heading}")' if heading else "",
                body=body[:8000],
                context=context,
            ),
            REVIEW_SCHEMA,
            system=REVIEW_SYSTEM,
            temperature=0.2,
            # Without thinking the reviewer passed an answer calling Raft "superior to Paxos"
            # for latency beside a passage saying their performance is similar; with it, it
            # flags that and the adopted design claims. About ten seconds on the 30B.
            think=True,
            num_predict=4000,
        )
    except Exception:
        log.warning("review failed for %r", heading or brief[:60], exc_info=True)
        return []
    flags = []
    for f in out.get("flags") or []:
        quote = " ".join(str(f.get("quote") or "").split())
        issue = " ".join(str(f.get("issue") or "").split())
        if quote and issue and not is_placeholder(quote) and anchored(quote, body):
            flags.append(
                {
                    "quote": clip(quote, 300),
                    "issue": clip(issue, 500),
                    "condition": clip(str(f.get("condition") or ""), 500),
                }
            )
    return flags
