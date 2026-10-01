"""Rows to passage text, by the manifest's rules alone.

A line format names fields, nested with dots and indexed with [n] (`{user.login}`,
`{labels[0].name}`), each optionally through filters:

    {created_at|date}       2026-09-27 14:03 from any ISO timestamp (or epoch seconds)
    {labels|join}           a list, joined with ", " (of `name` fields when they are objects)
    {body|trunc:80}         cut to 80 characters, at a word
    {state|upper} {x|lower}
    {assignee|default:none} when missing or empty
    {items|count}           how many

A missing field prints blank rather than raising, so a live shape that drifts a little
still renders."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from library_agent.modules.execute import dig
from library_agent.modules.manifest import Operation

_FIELD = re.compile(r"\{([^{}|]+)((?:\|[^{}|]+)*)\}")


def _date(v: Any) -> str:
    if v in (None, ""):
        return ""
    try:
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
            n = float(v)
            dt = datetime.fromtimestamp(n / 1000 if n > 1e11 else n, UTC)
        else:
            dt = datetime.fromisoformat(str(v))
        return dt.strftime("%Y-%m-%d %H:%M") if (dt.hour or dt.minute) else dt.strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return str(v)


def _text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, list):
        return ", ".join(_text(x.get("name", x) if isinstance(x, dict) else x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)[:200]
    return str(v)


def _apply(v: Any, flt: str) -> Any:
    name, _, arg = flt.partition(":")
    name = name.strip()
    if name == "date":
        return _date(v)
    if name == "join":
        return _text(v if isinstance(v, list) else [v] if v not in (None, "") else [])
    if name == "trunc":
        s, n = _text(v), int(arg or 80)
        if len(s) <= n:
            return s
        cut = s[:n].rsplit(" ", 1)[0] or s[:n]
        return cut.rstrip(",.;: ") + "…"
    if name == "upper":
        return _text(v).upper()
    if name == "lower":
        return _text(v).lower()
    if name == "default":
        return v if v not in (None, "", []) else arg
    if name == "count":
        return len(v) if isinstance(v, (list, dict, str)) else 0
    if name == "json":
        return json.dumps(v, ensure_ascii=False)[:300]
    return v


def format_row(line: str, row: Any) -> str:
    """One row through the line format."""
    base = row if isinstance(row, dict) else {"value": row}

    def one(m: re.Match) -> str:
        path, filters = m.group(1).strip(), m.group(2)
        v = base.get(path) if path in base else dig(base, path)
        if path == "value" and not isinstance(row, dict):
            v = row
        for flt in [f for f in filters.split("|") if f]:
            v = _apply(v, flt)
        return _text(v)

    return " ".join(_FIELD.sub(one, line).split())


def render_rows(op: Operation, data: Any) -> str:
    """The operation's rendering of a response: a line per row, capped, or the empty line
    when there is nothing. `data` is the rows a call returned, or a whole JSON document."""
    r = op.render
    if isinstance(data, list):
        rows = data
    else:
        rows = dig(data, r.rows)
        rows = [rows] if isinstance(rows, dict) else rows if isinstance(rows, list) else []
    if not rows:
        return r.empty
    head = summarise(r.summary, rows) if r.summary else ""
    lines = [ln for ln in (format_row(r.line, row) for row in rows[: r.limit]) if ln.strip()]
    more = len(rows) - r.limit
    text = "\n".join(lines)
    if more > 0:
        text += f"\n… and {more} more"
    if head:
        text = f"{head}\n{text}"
    return text or r.empty


_SUMMARY = re.compile(r"\{(count|distinct:[A-Za-z0-9_.\[\]*]+)\}")


def summarise(template: str, rows: list) -> str:
    """{count}: the rows fetched; {distinct:path}: how many different values that field
    takes across them (empty values not counted)."""

    def one(m: re.Match) -> str:
        key = m.group(1)
        if key == "count":
            return str(len(rows))
        path = key.split(":", 1)[1]
        vals = {
            json.dumps(v, sort_keys=True)
            for row in rows
            if (v := dig(row, path)) not in (None, "", [])
        }
        return str(len(vals))

    return _SUMMARY.sub(one, template)
