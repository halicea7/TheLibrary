"""The consult step: between rewriting the question and retrieving from the shelf, the
librarian may put a question to a seated module. It is one schema-constrained call — the
enum is the seated operations, and the model fills that operation's declared parameters, or
picks 'none', which is the common case and costs one small call. The model never writes a
URL or code; it names an operation and fills fields. The chosen operation runs, its JSON is
rendered to a passage, and that passage joins the pool the answer is built and verified
from — cited like any other, but marked as live."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from library_agent.modules.execute import ModuleError, call
from library_agent.modules.manifest import Module, Operation
from library_agent.modules.render import render_rows
from library_agent.modules.store import ModuleConfig

log = logging.getLogger(__name__)

# Measured on ten questions (six about the fleet, four about documents): telling the model
# "none is the normal case" and asking for the operation straight away got 6/10 -- it
# answered none to "How many endpoints have Zoom installed?" and "How many agents do we
# have?". Letting it say what the question needs first, and saying when a live look *is*
# wanted, got 10/10 with no false positives on the document questions.
CONSULT_SYSTEM = (
    "You decide whether a reader's question needs a live look at a connected system, and if "
    "so, which one operation. Most questions are about the library's documents and need none. "
    "But when the question asks about the reader's own systems -- what is installed, where, "
    "how many, which hosts, whether something was seen -- and an operation below answers "
    "exactly that, choose it. Fill parameters only from the question; never guess an "
    "identifier that is not in it."
)

CONSULT_PROMPT = """A reader asked the library:
{question}

You may consult one live operation, or none. The operations available:
{operations}

First say in `need` what the question asks for. Then choose exactly one `operation` id from
the list, or "none". If you choose one, put its parameters in `parameters`, taking the values
from the question only."""


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
            # First and required, so the model says what the question needs before it
            # commits. Optional, it is skipped -- and the choice fell to 4/10 (all none).
            "need": {"type": "string", "maxLength": 200},
            "operation": {"type": "string", "enum": [*op_ids, "none"]},
            "parameters": {"type": "object"},
        },
        "required": ["need", "operation"],
    }


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
    # An exact-match input is never taken from the question (see manifest._param).
    given = {k: v for k, v in given.items() if not (op.param_specs.get(k) or {}).get("exact")}
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
