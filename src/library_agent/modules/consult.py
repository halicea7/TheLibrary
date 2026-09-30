"""The consult step: between rewriting the question and retrieving from the shelf, the
librarian may put a question to a seated module. It is one schema-constrained call — the
enum is the seated operations, and the model fills that operation's declared parameters, or
picks 'none', which is the common case and costs one small call. The model never writes a
URL or code; it names an operation and fills fields. The chosen operation runs, its JSON is
rendered to a passage, and that passage joins the pool the answer is built and verified
from — cited like any other, but marked as live."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from library_agent.modules.execute import ModuleError, call
from library_agent.modules.manifest import Module, Operation
from library_agent.modules.render import render_rows
from library_agent.modules.store import ModuleConfig

log = logging.getLogger(__name__)

# Routing, measured on 17 questions with SentinelOne and NVD seated (9 about the fleet, 3
# about CVEs, 5 about documents). Asking for the operation outright, with "none is the
# normal case", got 6/10 on the first ten; saying what the question needs first, 10/10 --
# until two more operations were added, when it fell to 9/14: the model echoed the
# question as its "need" and chose none. Naming the closest operation first, then the one
# to run, got 17/17. A value the model invents for an input ("BERT" as a CVE id) is caught
# by the input's pattern, not by the prompt.
CONSULT_SYSTEM = (
    "You route a reader's question. Some questions are about the library's documents -- what "
    "something is, how it works, what a paper or runbook says. Others ask about the reader's "
    "own live systems -- what is installed, what happened recently, what was detected, how "
    "many hosts or agents, whether something was seen. For those, a connected system's "
    "operation answers it. Fill parameters only from the question; never guess an identifier "
    "that is not in it."
)

CONSULT_PROMPT = """A reader asked:
{question}

Connected systems and what each operation answers:
{operations}

Decide in order:
1. `closest`: the operation that fits the question best (always name one).
2. `operation`: the operation to run -- normally `closest` when the question is about the
   reader's own systems; "none" when it is about the library's documents or no operation
   truly answers it.
Put any parameters in `parameters`, from the question only."""


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
            # First and required: naming the nearest operation before deciding is what
            # stopped the model defaulting to none (see CONSULT_SYSTEM).
            "closest": {"type": "string", "enum": op_ids},
            "operation": {"type": "string", "enum": [*op_ids, "none"]},
            "parameters": {"type": "object"},
        },
        "required": ["closest", "operation"],
    }


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
) -> list[LiveResult]:
    """Consult the seated modules for this question. Returns at most one live result
    (including a result that is an error, so the answer can say the source was down rather
    than thinning silently)."""
    ops = [(m, c, op) for (m, c) in seated for op in m.operations]
    if not ops:
        return []
    op_ids = [f"{m.id}.{op.id}" for (m, _c, op) in ops]
    lines = "\n".join(
        f"- {m.id}.{op.id}: {op.summary} Use when {op.ask_when} "
        f"Parameters: {', '.join((op.params.get('properties') or {}).keys()) or 'none'}"
        for (m, _c, op) in ops
    )
    try:
        out = await client.structured(
            model,
            CONSULT_PROMPT.format(question=question, operations=lines),
            _schema(op_ids),
            system=CONSULT_SYSTEM,
            think=False,
            temperature=0.0,
            num_predict=300,
        )
    except Exception:
        log.warning("module consult failed", exc_info=True)
        return []
    choice = str(out.get("operation") or "none")
    if choice == "none":
        return []
    match = next(((m, c, op) for (m, c, op) in ops if f"{m.id}.{op.id}" == choice), None)
    if not match:
        return []
    module, cfg, op = match
    given = out.get("parameters") if isinstance(out.get("parameters"), dict) else {}
    given = {k: v for k, v in given.items() if v not in (None, "")}
    # An exact-match input is never taken from the question, and a value that doesn't look
    # like what the input takes (a CVE id that isn't one) is dropped (see manifest._param).
    given = {
        k: v
        for k, v in given.items()
        if not (op.param_specs.get(k) or {}).get("exact")
        and _looks_right(v, (op.param_specs.get(k) or {}).get("pattern"))
    }
    op = _fillable(op, given, [o for (m, _c, o) in ops if m is module])
    if op is None:
        return []  # nothing the question gives can fill any fitting operation
    args = {k: given[k] for k in (op.params.get("properties") or {}) if k in given}
    import time

    try:
        res = await call(module, cfg, op, args)
    except ModuleError as exc:
        return [LiveResult(module=module, op=op, args=args, when=time.time(), error=str(exc))]
    return [
        LiveResult(
            module=module,
            op=op,
            args=args,
            when=res["when"],
            count=res["count"],
            text=render_rows(op, res["rows"]),
        )
    ]
