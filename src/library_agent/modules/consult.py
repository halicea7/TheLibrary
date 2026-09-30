"""The consult step: between rewriting the question and retrieving from the shelf, the
librarian may put a question to a seated module. It is one schema-constrained call — the
enum is the seated operations, and the model fills that operation's declared parameters, or
picks 'none', which is the common case and costs one small call. The model never writes a
URL or code; it names an operation and fills fields. The chosen operation runs, its JSON is
rendered to a passage, and that passage joins the pool the answer is built and verified
from — cited like any other, but marked as live."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from library_agent.chat.grounding import ungrounded
from library_agent.modules.execute import ModuleError, call
from library_agent.modules.manifest import Module, Operation
from library_agent.modules.render import render_rows
from library_agent.modules.store import ModuleConfig

log = logging.getLogger(__name__)

# How the librarian uses modules. It used to make one snap choice -- one operation, no
# thinking, the bare question -- which was measured and tuned (6/10 -> 17/17 on routing)
# but couldn't follow a conversation ("look up those CVEs you just gave me") or combine
# sources ("how severe is it, and are we exposed?"). Now it thinks, may call several
# operations across modules, sees what came back, and may call again: at most
# MAX_ROUNDS rounds and MAX_CALLS calls a question. Whatever it proposes still passes the
# same checks in code: exact inputs are never taken from it, a value must fit its
# input's pattern, and an identifier (CVE, hash, IP) must already be in the question,
# the conversation or an earlier result -- a model that can't see the ids invents them.
MAX_ROUNDS = 3
MAX_CALLS = 4
PER_ROUND = 3

CONSULT_SYSTEM = (
    "You decide which of the reader's connected systems to query, if any, to answer their "
    "question. Some questions are about the library's documents -- what something is, how it "
    "works, what a paper or runbook says: those need no call. Others ask about the reader's own "
    "live systems or about current data a connected service holds: what is installed, what "
    "happened recently, what was detected, how many hosts, whether something was seen, how "
    "severe a CVE is. Think it through, then call what answers it -- several operations if the "
    "question has several parts. Take every identifier (CVE ids, hashes, addresses, names) from "
    "the question, the conversation or earlier results; never make one up. If an operation "
    "can't be filled from what you have, don't call it."
)

CONSULT_PROMPT = """Today is {today}.
{conversation}The reader now asks:
{question}

Connected systems and their operations:
{operations}
{results}
Decide what to call next. List the calls in `calls` (at most {room}), each an operation and
its parameters. When nothing more is needed -- or nothing was needed at all -- return no
calls and set `done` to true."""


@dataclass
class LiveResult:
    module: Module
    op: Operation
    args: dict[str, Any]
    when: float
    text: str = ""
    count: int = 0
    error: str = ""
    chunk_id: uuid.UUID = field(default_factory=uuid.uuid4)

    def label(self) -> str:
        t = datetime.fromtimestamp(self.when, UTC).strftime("%H:%M")
        arg = next(iter(self.args.values()), "") if self.args else ""
        return f"{self.module.name} · {self.op.id}{f' · {arg}' if arg else ''} · {t}"


def _schema(op_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "calls": {
                "type": "array",
                "maxItems": PER_ROUND,
                "items": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": op_ids},
                        "parameters": {"type": "object"},
                    },
                    "required": ["operation"],
                },
            },
            "done": {"type": "boolean"},
        },
        "required": ["calls", "done"],
    }


def _catalogue(ops) -> str:
    out = []
    for m, _c, op in ops:
        params = []
        for name, schema in (op.params.get("properties") or {}).items():
            spec = op.param_specs.get(name) or {}
            bits = [name]
            if name in (op.params.get("required") or []):
                bits.append("required")
            if spec.get("pattern"):
                bits.append(f"like {spec['pattern']}")
            if spec.get("exact"):
                bits.append("exact value from the service's own listing; leave out")
            desc = schema.get("description")
            params.append(
                f"{' '.join(bits[:1])} ({', '.join(bits[1:]) or 'optional'}){': ' + desc if desc else ''}"
            )
        out.append(
            f"- {m.id}.{op.id} [{m.name}]: {op.summary} Use when {op.ask_when}"
            f"\n    parameters: {'; '.join(params) or 'none'}"
        )
    return "\n".join(out)


def _conversation(history: list[tuple[str, str]] | None) -> str:
    """The recent turns, the last answer at length: a follow-up points into it."""
    turns = (history or [])[-4:]
    if not turns:
        return ""
    lines = [
        f"{role}: {text[:3000] if i == len(turns) - 1 else text[:600]}"
        for i, (role, text) in enumerate(turns)
    ]
    return "The conversation so far:\n" + "\n".join(lines) + "\n\n"


def _results_block(results: list[LiveResult]) -> str:
    if not results:
        return ""
    parts = ["\nWhat the calls so far returned:"]
    for r in results:
        body = (
            f"error: {r.error}" if r.error else "\n".join(r.text.splitlines()[:12]) or "(no rows)"
        )
        parts.append(f"- {r.module.id}.{r.op.id}({json.dumps(r.args)}):\n{body}")
    return "\n".join(parts) + "\n"


def _prepare(call_spec: dict, ops, known: str):
    """A proposed call, checked: (module, cfg, op, args), or None if it can't be made
    honestly from what the reader and the results have given."""
    choice = str(call_spec.get("operation") or "")
    match = next(((m, c, op) for (m, c, op) in ops if f"{m.id}.{op.id}" == choice), None)
    if not match:
        return None
    module, cfg, op = match
    given = call_spec.get("parameters") if isinstance(call_spec.get("parameters"), dict) else {}
    given = {k: v for k, v in given.items() if v not in (None, "")}
    # An exact-match input is never taken from the model; a value must fit its input's
    # pattern; an identifier must already be in the known text (see grounding).
    given = {
        k: v
        for k, v in given.items()
        if not (op.param_specs.get(k) or {}).get("exact")
        and _looks_right(v, (op.param_specs.get(k) or {}).get("pattern"))
        and not ungrounded(str(v), known)
    }
    op = _fillable(op, given, [o for (m, _c, o) in ops if m is module])
    if op is None:
        return None
    args = {k: given[k] for k in (op.params.get("properties") or {}) if k in given}
    return module, cfg, op, args


async def _run(module, cfg, op, args) -> LiveResult:
    try:
        res = await call(module, cfg, op, args)
    except ModuleError as exc:
        return LiveResult(module=module, op=op, args=args, when=time.time(), error=str(exc))
    return LiveResult(
        module=module,
        op=op,
        args=args,
        when=res["when"],
        count=res["count"],
        text=render_rows(op, res["rows"]),
    )


def _looks_right(value: Any, pattern: str | None) -> bool:
    if not pattern:
        return True
    return bool(re.fullmatch(pattern, str(value).strip(), flags=re.IGNORECASE))


def _required(op: Operation) -> list[str]:
    return list(op.params.get("required") or [])


def _fillable(op: Operation, given: dict, siblings: list[Operation]) -> Operation | None:
    """The chosen operation if the question fills its required inputs; otherwise the
    nearest one of the same module it does fill -- one taking the inputs given (e.g. "which
    hosts have Chrome?" picks the per-host lookup, which also needs the vendor; the
    inventory lookup takes just the name and still answers). None if there is none."""
    if all(k in given for k in _required(op)):
        return op
    for o in siblings:
        req = _required(o)
        if o is not op and req and all(k in given for k in req):
            return o
    return None


async def consult(
    client,
    model: str,
    question: str,
    seated: list[tuple[Module, ModuleConfig]],
    history: list[tuple[str, str]] | None = None,
    *,
    on_call=None,
) -> list[LiveResult]:
    """Query the seated modules for this question: think, call what answers it (several
    operations if it has several parts), look at what came back, call again if needed.
    Returns every result, errors included, so the answer can say a source was down rather
    than thinning silently. `on_call(module, op, args)` is told as each call starts."""
    ops = [(m, c, op) for (m, c) in seated for op in m.operations]
    if not ops:
        return []
    op_ids = [f"{m.id}.{op.id}" for (m, _c, op) in ops]
    catalogue = _catalogue(ops)
    conversation = _conversation(history)
    known = question + "\n" + "\n".join(text for _, text in (history or []))
    results: list[LiveResult] = []
    made: set[str] = set()
    for _round in range(MAX_ROUNDS):
        room = min(PER_ROUND, MAX_CALLS - len(results))
        if room <= 0:
            break
        try:
            out = await client.structured(
                model,
                CONSULT_PROMPT.format(
                    today=datetime.now(UTC).date().isoformat(),
                    conversation=conversation,
                    question=question,
                    operations=catalogue,
                    results=_results_block(results),
                    room=room,
                ),
                _schema(op_ids),
                system=CONSULT_SYSTEM,
                think=True,
                temperature=0.0,
                num_predict=900,
            )
        except Exception:
            log.warning("module consult failed", exc_info=True)
            break
        batch = []
        for spec in (out.get("calls") or [])[:room]:
            prepared = _prepare(spec if isinstance(spec, dict) else {}, ops, known)
            if not prepared:
                continue
            module, _cfg, op, args = prepared
            key = f"{module.id}.{op.id}:{json.dumps(args, sort_keys=True)}"
            if key in made:
                continue
            made.add(key)
            batch.append(prepared)
        if not batch:
            break
        for module, _cfg, op, args in batch:
            if on_call:
                await on_call(module, op, args)
        got = await asyncio.gather(*(_run(*b) for b in batch))
        results.extend(got)
        # What came back may name what to look up next (a product, a CVE, a host).
        known += "\n" + "\n".join(r.text for r in got)
        if out.get("done"):
            break
    return results
