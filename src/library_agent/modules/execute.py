"""Calling a module's operation, safely. The URL is built from the manifest's path template
and the declared parameters, never from anything the model wrote. It is a GET, to the host
the operator pinned; a redirect to any other host is refused; the token is attached by the
manifest's scheme. The JSON that comes back is rendered to passage text by the manifest's
own rules — no second model call, so a citation traces to the bytes."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

import httpx

from library_agent.modules.manifest import Module, Operation
from library_agent.modules.store import ModuleConfig

# (module id, op id, frozenset(params)) -> (when, rendered text, count). A live call is
# repeated within a conversation; a brief cache spares the tenant and keeps an answer
# consistent with its follow-ups.
_cache: dict[tuple, tuple[float, str, int, dict]] = {}
CACHE_TTL = 90.0


class ModuleError(Exception):
    pass


def _params_to_query(op: Operation, args: dict[str, Any]) -> dict[str, str]:
    q: dict[str, str] = dict(op.const_query)
    for name, key in op.query.items():
        if name in args and args[name] not in (None, ""):
            q[key] = str(args[name])
    return q


async def call(module: Module, cfg: ModuleConfig, op: Operation, args: dict[str, Any]) -> dict:
    """Run one operation. Returns {ok, rows, count, url, when, error}."""
    key = (module.id, op.id, tuple(sorted((k, str(v)) for k, v in args.items())))
    hit = _cache.get(key)
    now = time.time()
    if hit and now - hit[0] < CACHE_TTL:
        return {"ok": True, "rows": hit[3], "count": hit[2], "when": hit[0], "cached": True}

    base = cfg.base_url.rstrip("/")
    host = urlparse(base).hostname
    if not host:
        raise ModuleError(f"{module.name}: no base URL configured")
    query = _params_to_query(op, args)
    headers = {module.auth_header: f"{module.token_prefix()}{cfg.token}", **cfg.extra_headers}

    # GET only, no automatic redirects (a 3xx to another host must not carry the token).
    async with httpx.AsyncClient(base_url=base, timeout=30.0, follow_redirects=False) as client:
        try:
            resp = await client.get(op.path, params=query, headers=headers)
        except httpx.HTTPError as exc:
            raise ModuleError(f"{module.name}: {type(exc).__name__}") from exc
        if resp.is_redirect:
            loc = resp.headers.get("location", "")
            if urlparse(loc).hostname not in (None, host):
                raise ModuleError(f"{module.name}: refused a redirect off {host}")
            raise ModuleError(f"{module.name}: unexpected redirect")
        if resp.status_code == 401 or resp.status_code == 403:
            raise ModuleError(f"{module.name}: not authorised ({resp.status_code})")
        if resp.status_code >= 400:
            raise ModuleError(f"{module.name}: HTTP {resp.status_code}")
        data = resp.json()

    rows = _dig(data, op.render.rows)
    rows = rows if isinstance(rows, list) else []
    _cache[key] = (now, "", len(rows), data)
    return {"ok": True, "rows": data, "count": len(rows), "when": now, "cached": False}


def _dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur
