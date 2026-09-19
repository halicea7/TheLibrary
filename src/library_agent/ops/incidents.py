"""Incidents: what went wrong, and the library's own advice on fixing it.

Recording is deliberately dumb and cheap -- a row per distinct error, a count when it
recurs -- and it is fed from four places: the API's unhandled-exception handler, a failed
chat turn, a failed background job, and a logging handler that catches anything logged at
ERROR with a traceback. Nothing here ever tries to fix anything.

Troubleshooting is the model reading the incident against *our own documentation* --
README, DEVLOG, TROUBLESHOOTING, the launcher and ops scripts -- and writing back a
diagnosis, steps with commands for the person to run themselves, the passages it leaned
on, and, when the problem looks like a defect rather than an environment issue, a
ready-made GitHub issue. It says what to do; it does not do it."""

from __future__ import annotations

import asyncio
import logging
import re
import traceback
import urllib.parse
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import Incident
from library_agent.db.session import session_scope
from library_agent.llm.ollama import Ollama

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]

# What the model may read. Order is preference when excerpts tie.
DOC_SOURCES = (
    "docs/TROUBLESHOOTING.md",
    "README.md",
    "docs/DEVLOG.md",
    "library",
    "scripts/bootstrap.sh",
    "ops/install.sh",
    "ops/uninstall.sh",
    "ops/backup.sh",
)


# --------------------------------------------------------------------------- recording


def _normalise(message: str) -> str:
    """Strip the parts of a message that vary between identical failures (ids, numbers,
    paths, quoted strings) so recurrences collapse onto one incident."""
    m = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<id>", message)
    m = re.sub(r"/[\w./-]+", "<path>", m)
    m = re.sub(r"'[^']{0,80}'", "'…'", m)
    m = re.sub(r"\d+", "N", m)
    return m[:400]


async def record(
    db: AsyncSession,
    *,
    source: str,
    kind: str,
    message: str,
    detail: str | None = None,
    context: dict[str, Any] | None = None,
) -> Incident:
    window = timedelta(seconds=settings().incident_dedupe_seconds)
    since = datetime.now(UTC) - window
    key = _normalise(message)
    recent = list(
        (
            await db.execute(
                select(Incident).where(
                    Incident.source == source,
                    Incident.kind == kind,
                    Incident.resolved.is_(False),
                    Incident.last_at >= since,
                )
            )
        ).scalars()
    )
    for inc in recent:
        if _normalise(inc.message) == key:
            inc.count += 1
            inc.last_at = datetime.now(UTC)
            inc.message = message[:2000]
            if detail:
                inc.detail = detail[-8000:]
            if context:
                inc.context = {**(inc.context or {}), **context}
            await db.flush()
            return inc
    inc = Incident(
        source=source,
        kind=kind[:80],
        message=message[:2000],
        detail=detail[-8000:] if detail else None,
        context=context,
    )
    db.add(inc)
    await db.flush()
    return inc


async def record_exception(
    exc: BaseException, *, source: str, context: dict[str, Any] | None = None
) -> None:
    """Fire-and-forget from any failure path. Never raises: an error in error recording
    must not mask the original."""
    try:
        async with session_scope() as db:
            await record(
                db,
                source=source,
                kind=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
                detail="".join(traceback.format_exception(exc)),
                context=context,
            )
    except Exception:
        log.debug("could not record incident", exc_info=True)


class IncidentHandler(logging.Handler):
    """Logging handler: anything at ERROR or above from our own loggers becomes an
    incident. Scheduled onto the running loop so the log call itself stays synchronous."""

    def __init__(self, source: str) -> None:
        super().__init__(level=logging.ERROR)
        self.source = source

    def emit(self, rec: logging.LogRecord) -> None:
        if rec.name.startswith("library_agent.ops"):
            return  # never record our own recording
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        message = rec.getMessage()
        detail = "".join(traceback.format_exception(*rec.exc_info)) if rec.exc_info else None
        kind = (
            rec.exc_info[0].__name__
            if rec.exc_info and rec.exc_info[0]
            else rec.name.rsplit(".", 1)[-1]
        )
        context = {"logger": rec.name, "where": f"{rec.module}.{rec.funcName}:{rec.lineno}"}

        async def go() -> None:
            try:
                async with session_scope() as db:
                    await record(
                        db,
                        source=self.source,
                        kind=kind,
                        message=message,
                        detail=detail,
                        context=context,
                    )
            except Exception:  # noqa: BLE001 - recording must never itself raise or log at ERROR
                return

        loop.create_task(go())


def install_handler(source: str) -> None:
    root = logging.getLogger("library_agent")
    if not any(isinstance(h, IncidentHandler) for h in root.handlers):
        root.addHandler(IncidentHandler(source))


# --------------------------------------------------------------------------- listing


async def list_incidents(db: AsyncSession, *, include_resolved: bool = False) -> list[Incident]:
    q = select(Incident).order_by(Incident.last_at.desc()).limit(200)
    if not include_resolved:
        q = q.where(Incident.resolved.is_(False))
    return list((await db.execute(q)).scalars())


async def resolve(db: AsyncSession, incident_id: uuid.UUID, *, resolved: bool = True) -> None:
    await db.execute(update(Incident).where(Incident.id == incident_id).values(resolved=resolved))


async def open_count(db: AsyncSession) -> int:
    return (await db.execute(select(func.count()).where(Incident.resolved.is_(False)))).scalar_one()


# --------------------------------------------------------------------------- documentation


def _paragraphs(path: Path) -> list[tuple[str, str]]:
    """(heading trail, paragraph) pairs. Shell scripts are read as comment blocks and the
    commands that follow them, so `./library status` shows up as a thing one can run."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rel = str(path.relative_to(ROOT))
    out: list[tuple[str, str]] = []
    if path.suffix == ".md":
        trail: list[str] = []
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if not block:
                continue
            if m := re.match(r"^(#{1,4})\s+(.*)", block):
                level = len(m.group(1))
                trail = trail[: level - 1] + [m.group(2).strip()]
                rest = block[m.end() :].strip()
                if rest:
                    out.append((f"{rel} › {' › '.join(trail)}", rest))
                continue
            out.append((f"{rel} › {' › '.join(trail)}" if trail else rel, block))
    else:
        # Scripts: header comment as one excerpt, then each comment+code stanza.
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if len(block) > 40:
                out.append((rel, block))
    return [(h, p[:1400]) for h, p in out]


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9_./-]{2,}", s.lower()) if w not in _STOP}


_STOP = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "are",
    "was",
    "not",
    "but",
    "you",
    "your",
    "its",
    "has",
    "have",
    "into",
    "when",
    "then",
    "than",
    "which",
    "what",
    "will",
    "one",
    "all",
    "any",
    "can",
    "our",
    "out",
    "use",
    "used",
    "line",
    "file",
    "error",
}


def relevant_docs(query: str, *, limit: int = 8) -> list[tuple[str, str, float]]:
    """Keyword overlap, weighted toward rarer words and toward the troubleshooting guide.
    Embedding the docs would be more exact and is not worth a model call for a dozen
    files; this is a retrieval problem the size of a grep."""
    q = _tokens(query)
    if not q:
        return []
    paras: list[tuple[str, str]] = []
    for src in DOC_SOURCES:
        paras.extend(_paragraphs(ROOT / src))
    if not paras:
        return []
    df: dict[str, int] = {}
    toks = [_tokens(f"{h} {p}") for h, p in paras]
    for ts in toks:
        for w in ts:
            df[w] = df.get(w, 0) + 1
    n = len(paras)
    scored = []
    for (h, p), ts in zip(paras, toks, strict=True):
        hit = q & ts
        if not hit:
            continue
        score = sum(1.0 / (1 + df[w] / n * 10) for w in hit)
        if h.startswith("docs/TROUBLESHOOTING"):
            score *= 1.5
        scored.append((h, p, score))
    scored.sort(key=lambda x: -x[2])
    return scored[:limit]


# --------------------------------------------------------------------------- commands

# Tools whose first word is enough to trust: they are general and documented elsewhere.
_GENERIC = {
    "curl",
    "psql",
    "redis-cli",
    "pg_isready",
    "ps",
    "ls",
    "grep",
    "tail",
    "cat",
    "brew",
    "createdb",
    "ssh",
    "lsof",
    "pgrep",
    "open",
    "cd",
    "echo",
    "uv",
    "python3",
    "git",
}


def documented_commands() -> set[str]:
    """Every command our docs and scripts show: lines in fenced shell blocks, inline code
    spans that look like commands, and the usage lines of the launcher. A suggested
    command is trusted when its first two words match one of these."""
    cmds: set[str] = set()
    # Not the devlog: it narrates what went wrong, including commands the model once
    # invented, and quoting one there must not make it real.
    for src in (s for s in DOC_SOURCES if not s.endswith("DEVLOG.md")):
        path = ROOT / src
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for block in re.findall(r"```(?:sh|bash|zsh)?\n(.*?)```", text, re.DOTALL):
            for line in block.splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    cmds.add(line)
        for span in re.findall(r"`([^`\n]{3,120})`", text):
            if re.match(
                r"^(\./library|uv run|curl|psql|redis-cli|pg_isready|brew|createdb|alembic)\b", span
            ):
                cmds.add(span.strip())
        if path.name == "library":
            for m in re.finditer(r"^#\s+(\./library\s+\S+.*)$", text, re.MULTILINE):
                cmds.add(m.group(1).split("  ")[0].strip())
            for m in re.finditer(r"^\s+([a-z|]+)\)", text, re.MULTILINE):
                for sub in m.group(1).split("|"):
                    cmds.add(f"./library {sub}")
    return cmds


def command_is_documented(command: str, known: set[str] | None = None) -> bool:
    known = documented_commands() if known is None else known
    words = command.strip().split()
    if not words:
        return False
    if words[0] in _GENERIC and words[0] != "uv":
        return True
    head2 = " ".join(words[:2])
    head3 = " ".join(words[:3])
    for k in known:
        kw = k.split()
        if " ".join(kw[:2]) == head2 and (
            len(kw) < 3
            or len(words) < 3
            or " ".join(kw[:3]) == head3
            or not kw[2].startswith("-")
            and not words[2].startswith("-")
        ):
            return True
    return False


# --------------------------------------------------------------------------- advice

ADVICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string", "maxLength": 900},
        "likely_cause": {
            "type": "string",
            "enum": ["environment", "configuration", "data", "model", "defect", "unknown"],
        },
        "steps": {
            "type": "array",
            "maxItems": 7,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "maxLength": 120},
                    "command": {"type": "string", "maxLength": 300},
                    "why": {"type": "string", "maxLength": 300},
                },
                "required": ["title", "why"],
            },
        },
        "docs_used": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 120},
        },
        "open_issue": {"type": "boolean"},
        "issue_title": {"type": "string", "maxLength": 120},
        "issue_body": {"type": "string", "maxLength": 2500},
    },
    "required": ["diagnosis", "likely_cause", "steps", "docs_used", "open_issue"],
}

ADVICE_PROMPT = """You are the librarian of a self-hosted research library, helping its owner
fix a problem with it. You do not run anything yourself. You read the incident and the
library's own documentation below, and you tell the owner what is going on and what to do.

Rules:
- Base yourself on the documentation. Cite which excerpts you used in docs_used by their
  headings. If the documentation does not cover it, say so plainly in the diagnosis.
- Steps are for the owner to run by hand. Give the exact command when there is one
  (`./library status`, `redis-cli ping`, `uv run alembic upgrade head`, ...). Never invent
  a flag or a script that the documentation does not show.
- Ollama at localhost:11434 is a remote GPU box reached over an SSH tunnel. Never suggest
  starting, stopping or killing a local Ollama; suggest checking the tunnel.
- Say whether this looks like the environment (a service down, a tunnel dropped), the
  configuration, the data (a bad file), the model (a bad or empty generation), or a
  defect in the library's own code. Set open_issue true only for a defect or for
  something the documentation cannot explain after the steps. When you do, write
  issue_title and an issue_body in markdown with: what happened, the error, what was
  tried, and the environment; leave out anything that looks like a secret or a home path.

Incident
--------
source: {source}
kind: {kind}
seen: {count} time(s), last {last_at}
message: {message}
context: {context}

traceback (tail):
{detail}

Documentation excerpts
----------------------
{docs}
"""


def _issue_url(title: str, body: str) -> str:
    base = settings().repo_url.rstrip("/")
    q = urllib.parse.urlencode({"title": title[:120], "body": body[:6000], "labels": "incident"})
    return f"{base}/issues/new?{q}"


def _scrub(s: str) -> str:
    """Home paths and anything key-shaped stay out of issue text and prompts alike."""
    s = re.sub(r"/Users/[^/\s]+", "~", s)
    s = re.sub(r"/home/[^/\s]+", "~", s)
    s = re.sub(r"(?i)(api[_-]?key|token|password|secret)\s*[=:]\s*\S+", r"\1=<redacted>", s)
    return s


async def advise(db: AsyncSession, incident_id: uuid.UUID, *, client: Ollama | None = None) -> dict:
    inc = await db.get(Incident, incident_id)
    if not inc:
        raise KeyError("no such incident")
    cfg = settings()
    query = f"{inc.kind} {inc.message} {inc.source} " + (inc.detail or "")[-600:]
    docs = relevant_docs(query)
    doc_text = "\n\n".join(f"[{h}]\n{p}" for h, p, _ in docs) or "(nothing relevant found)"
    model = cfg.troubleshoot_model or cfg.chat_model_options.get("technical") or cfg.reader_model
    own = client is None
    c = client or Ollama()
    try:
        out = await c.structured(
            model,
            ADVICE_PROMPT.format(
                source=inc.source,
                kind=inc.kind,
                count=inc.count,
                last_at=inc.last_at.isoformat(timespec="minutes") if inc.last_at else "",
                message=_scrub(inc.message),
                context=_scrub(str(inc.context or {}))[:600],
                detail=_scrub((inc.detail or "(none)")[-2500:]),
                docs=_scrub(doc_text),
            ),
            ADVICE_SCHEMA,
            system="You are careful, concrete and brief. You never claim a step was done.",
            instructions=ADVICE_PROMPT,
            temperature=0.2,
            think=True,
            num_predict=3000,
        )
    finally:
        if own:
            await c.aclose()
    known = documented_commands()
    steps = []
    for s in out.get("steps") or []:
        if not s.get("title"):
            continue
        cmd = str(s.get("command") or "").strip() or None
        steps.append(
            {
                "title": str(s.get("title") or "").strip(),
                "command": cmd,
                "why": str(s.get("why") or "").strip(),
                # The model was told not to invent flags or scripts. It does anyway;
                # the UI marks what our documentation does not show.
                "documented": command_is_documented(cmd, known) if cmd else None,
            }
        )
    advice = {
        "model": model,
        "diagnosis": str(out.get("diagnosis") or "").strip(),
        "likely_cause": out.get("likely_cause") or "unknown",
        "steps": steps,
        "docs_used": [str(x) for x in out.get("docs_used") or []][:6],
        "docs_offered": list(dict.fromkeys(h for h, _, _ in docs)),
        "open_issue": bool(out.get("open_issue")),
    }
    if advice["open_issue"]:
        title = str(out.get("issue_title") or f"{inc.kind}: {inc.message[:60]}")
        body = _scrub(str(out.get("issue_body") or ""))
        if not body:
            body = (
                f"## What happened\n{inc.message}\n\n## Error\n```\n{(inc.detail or '')[-1500:]}\n```\n\n"
                f"## Source\n{inc.source} · {inc.kind} · seen {inc.count} time(s)"
            )
            body = _scrub(body)
        advice["issue_title"] = title
        advice["issue_body"] = body
        advice["issue_url"] = _issue_url(title, body)
    inc.advice = advice
    inc.advice_at = datetime.now(UTC)
    await db.flush()
    return advice
