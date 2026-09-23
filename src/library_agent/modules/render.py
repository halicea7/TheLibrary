"""JSON to passage text, by the manifest's rules alone. A row that is missing a field the
line format names prints a blank there rather than raising, so a live shape that drifts a
little still renders."""

from __future__ import annotations

from typing import Any

from library_agent.modules.execute import _dig
from library_agent.modules.manifest import Operation


class _Blank(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _flatten(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {"value": row}
    out: dict[str, Any] = {}
    for k, v in row.items():
        out[k] = "" if v is None else v
    return out


def render_rows(op: Operation, data: dict) -> str:
    """The operation's rendering of one response: a line per row, capped, or the empty
    line when there is nothing."""
    r = op.render
    rows = _dig(data, r.rows)
    rows = rows if isinstance(rows, list) else []
    if not rows:
        return r.empty
    lines = []
    for row in rows[: r.limit]:
        try:
            lines.append(r.line.format_map(_Blank(_flatten(row))))
        except (ValueError, KeyError, IndexError):
            continue
    more = len(rows) - r.limit
    text = "\n".join(lines)
    if more > 0:
        text += f"\n… and {more} more"
    return text or r.empty
