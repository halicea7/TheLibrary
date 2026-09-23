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

CONSULT_SYSTEM = (
    "You decide whether a reader's question needs a live look at a connected system, and if "
    "so, which one operation. Picking 'none' is the normal case — choose an operation only "
    "when the question plainly calls for it and you can fill its required parameters from the "
    "question. Never guess an identifier that is not in the question."
)

CONSULT_PROMPT = """A reader asked the library:
{question}

You may consult one live operation, or none. The operations available:
{operations}

Choose exactly one `operation` id from the list, or "none". If you choose one, put its
parameters in `parameters`, taking the values from the question only."""


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
            "operation": {"type": "string", "enum": [*op_ids, "none"]},
            "parameters": {"type": "object"},
        },
        "required": ["operation"],
    }


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
    args = {
        k: given.get(k)
        for k in (op.params.get("properties") or {})
        if given.get(k) not in (None, "")
    }
    for req in op.params.get("required") or []:
        if not args.get(req):
            return []  # the model could not fill a required field; treat as no consult
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
