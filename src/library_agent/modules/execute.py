"""Calling a connector's operation, safely.

The request is built from the manifest -- path template, typed parameters, fixed query,
and for a read-only search a JSON body -- never from anything the model wrote. It goes
only to the host the operator pinned (or one the manifest names in allowed_hosts); a
redirect is never followed; the secret is attached by the manifest's auth. Pages are
followed up to a bound; the response is read up to a byte cap; calls are rate-limited
per connector and briefly cached. What comes back is parsed by the operation's declared
format into rows, which render.py turns into passage text by the manifest's own rules --
no second model call, so a citation traces to the bytes."""

from __future__ import annotations

import base64
import csv
import io
import json
import re
import time
from collections import deque
from datetime import date
from typing import Any
from urllib.parse import quote, urljoin, urlparse
from xml.etree.ElementTree import Element

import httpx
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from library_agent.modules.manifest import HARD_MAX_PAGES, Module, Operation
from library_agent.modules.store import ModuleConfig

# (module id, op id, args) -> (when, rows, count). A live call is repeated within a
# conversation; a brief cache spares the source and keeps follow-ups consistent.
_cache: dict[tuple, tuple[float, list, int]] = {}
CACHE_TTL = 90.0
_calls: dict[str, deque] = {}  # module id -> recent call times, for the rate limit
_oauth: dict[tuple, tuple[float, str]] = {}  # (module id, base, client id) -> (expiry, token)


class ModuleError(Exception):
    pass


# ------------------------------------------------------------------- parameters


def coerce(op: Operation, args: dict[str, Any]) -> dict[str, Any]:
    """Declared parameters only, each checked against its type; defaults filled. A value
    that does not fit is refused with a reason rather than sent."""
    out: dict[str, Any] = {}
    props = op.params.get("properties") or {}
    for name in props:
        spec = op.param_specs.get(name) or {}
        v = args.get(name)
        if v in (None, "") and spec.get("default") is not None:
            v = spec["default"]
        if v in (None, ""):
            continue
        t = spec.get("type") or {"integer": "int", "number": "number", "boolean": "bool"}.get(
            props[name].get("type"), "string"
        )
        try:
            if t == "int":
                v = int(v)
            elif t == "number":
                v = float(v)
            elif t == "bool":
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            elif t == "date":
                v = date.fromisoformat(str(v)[:10]).isoformat()
            else:
                v = str(v)
        except (TypeError, ValueError) as exc:
            raise ModuleError(f"{name}: not a valid {t}") from exc
        if t in ("int", "number"):
            if spec.get("min") is not None and v < spec["min"]:
                v = spec["min"]
            if spec.get("max") is not None and v > spec["max"]:
                v = spec["max"]
        values = spec.get("values") or props[name].get("enum")
        if values and v not in values:
            raise ModuleError(f"{name}: must be one of {', '.join(values)}")
        limit = spec.get("max_length") or props[name].get("maxLength")
        if isinstance(v, str) and limit and len(v) > limit:
            raise ModuleError(f"{name}: longer than {limit} characters")
        out[name] = v
    for req in op.params.get("required") or []:
        if req not in out:
            raise ModuleError(f"{req} is required")
    return out


def _render_value(v: Any) -> str:
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


def _fill_body(template: Any, args: dict[str, Any]) -> Any:
    """A JSON body template with "{param}" strings replaced by the typed value (a whole
    string that is exactly one placeholder takes the value's own type)."""
    if isinstance(template, dict):
        return {k: _fill_body(v, args) for k, v in template.items()}
    if isinstance(template, list):
        return [_fill_body(v, args) for v in template]
    if isinstance(template, str):
        m = re.fullmatch(r"\{(\w+)\}", template)
        if m:
            return args.get(m.group(1))
        return re.sub(r"\{(\w+)\}", lambda mm: _render_value(args.get(mm.group(1), "")), template)
    return template


def build(op: Operation, args: dict[str, Any]) -> tuple[str, dict[str, str], Any]:
    """(path, query, body) for one call. Path values are percent-encoded whole, so a value
    cannot add a segment or a query of its own."""
    path = op.path
    for name in op.path_params():
        path = path.replace("{" + name + "}", quote(_render_value(args[name]), safe=""))
    query: dict[str, str] = {k: relative_dates(v) for k, v in op.const_query.items()}
    body_args: dict[str, Any] = {}
    for name, v in args.items():
        spec = op.param_specs.get(name) or {}
        where = spec.get("in") or ("path" if name in op.path_params() else "query")
        if where == "query":
            query[spec.get("key") or op.query.get(name) or name] = _render_value(v)
        elif where == "body":
            body_args[name] = v
    body = _fill_body(op.body, {**args, **body_args}) if op.method == "POST" else None
    return path, query, body


# ------------------------------------------------------------------- guard rails


def _hosts(module: Module, base: str) -> set[str]:
    return {urlparse(base).hostname or ""} | set(module.allowed_hosts)


def _rate(module: Module) -> None:
    q = _calls.setdefault(module.id, deque())
    now = time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= module.limits.rate_per_minute:
        raise ModuleError(
            f"{module.name}: rate limit ({module.limits.rate_per_minute}/min) reached"
        )
    q.append(now)


async def _auth(client: httpx.AsyncClient, module: Module, cfg: ModuleConfig, base: str):
    """(headers, query) carrying the secret, by the manifest's scheme."""
    a = module.auth_spec()
    if a.type == "none":
        return {}, {}
    if not cfg.token:
        raise ModuleError(f"{module.name}: no credentials configured")
    if a.type == "bearer":
        return {a.header: f"Bearer {cfg.token}"}, {}
    if a.type == "header":
        return {a.header: f"{a.prefix}{cfg.token}"}, {}
    if a.type == "query":
        return {}, {a.param: cfg.token}
    if a.type == "basic":
        pair = base64.b64encode(f"{cfg.username}:{cfg.token}".encode()).decode()
        return {"Authorization": f"Basic {pair}"}, {}
    if a.type == "oauth2_client":
        key = (module.id, base, cfg.username)
        hit = _oauth.get(key)
        if hit and hit[0] > time.time():
            return {"Authorization": f"Bearer {hit[1]}"}, {}
        url = (
            urljoin(base + "/", a.token_url.lstrip("/"))
            if "://" not in a.token_url
            else a.token_url
        )
        if (urlparse(url).hostname or "") not in _hosts(module, base):
            raise ModuleError(f"{module.name}: token URL is off the pinned host")
        data = {"grant_type": "client_credentials", **({"scope": a.scope} if a.scope else {})}
        try:
            r = await client.post(url, data=data, auth=(cfg.username, cfg.token))
        except httpx.HTTPError as exc:
            raise ModuleError(
                f"{module.name}: token exchange failed ({type(exc).__name__})"
            ) from exc
        if r.status_code >= 400:
            raise ModuleError(f"{module.name}: token exchange refused ({r.status_code})")
        body = r.json()
        tok = body.get("access_token")
        if not tok:
            raise ModuleError(f"{module.name}: no access_token in the token response")
        _oauth[key] = (time.time() + max(30, int(body.get("expires_in", 3600)) - 30), tok)
        return {"Authorization": f"Bearer {tok}"}, {}
    raise ModuleError(f"{module.name}: unknown auth {a.type}")


async def _fetch(
    client, module: Module, method: str, url: str, **kw
) -> tuple[httpx.Response, bytes]:
    """One request, read up to the connector's byte cap, redirects refused."""
    try:
        async with client.stream(method, url, **kw) as resp:
            if resp.is_redirect:
                loc = resp.headers.get("location", "")
                host = urlparse(str(resp.request.url)).hostname
                if urlparse(loc).hostname not in (None, host):
                    raise ModuleError(f"{module.name}: refused a redirect off {host}")
                raise ModuleError(f"{module.name}: unexpected redirect")
            if resp.status_code in (401, 403):
                raise ModuleError(f"{module.name}: not authorised ({resp.status_code})")
            if resp.status_code == 429:
                raise ModuleError(f"{module.name}: the source is rate-limiting us (429)")
            if resp.status_code >= 400:
                raise ModuleError(f"{module.name}: HTTP {resp.status_code}")
            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                buf += chunk
                if len(buf) > module.limits.max_bytes:
                    raise ModuleError(
                        f"{module.name}: response over {module.limits.max_bytes:,} bytes"
                    )
            return resp, bytes(buf)
    except httpx.HTTPError as exc:
        raise ModuleError(f"{module.name}: {type(exc).__name__}") from exc


# ------------------------------------------------------------------- responses

_SEG = re.compile(r"([^.\[\]]+)|\[(\*|\d+)\]")


_RELDATE = re.compile(r"\{now(?:([+-])(\d{1,4})([dh]))?(?::([^{}]{1,40}))?\}")


def relative_dates(value: str) -> str:
    """A fixed query value may name a moment relative to now, formatted as the service
    wants it: "{now-7d:%Y-%m-%dT00:00:00.000}" is the start of the day a week ago (UTC),
    "{now:%Y-%m-%d}" today. So "recent" needs no input a model would have to compute."""
    from datetime import UTC, datetime, timedelta

    def one(m: re.Match) -> str:
        sign, n, unit, fmt = m.groups()
        t = datetime.now(UTC)
        if n:
            delta = timedelta(days=int(n)) if unit == "d" else timedelta(hours=int(n))
            t = t - delta if sign == "-" else t + delta
        return t.strftime(fmt or "%Y-%m-%dT%H:%M:%SZ")

    return _RELDATE.sub(one, str(value))


def dig(obj: Any, path: str) -> Any:
    """A JSONPath-lite lookup: "data", "data.items[*]", "results[0].hits[*].doc",
    "agent.os.name". `[*]` fans out over a list; an empty path is the whole value."""
    if not path:
        return obj
    cur: list[Any] = [obj]
    fanned = False
    for name, idx in _SEG.findall(path):
        nxt: list[Any] = []
        for c in cur:
            if name:
                if isinstance(c, dict) and name in c:
                    nxt.append(c[name])
            elif idx == "*":
                fanned = True
                if isinstance(c, list):
                    nxt.extend(c)
            elif isinstance(c, list) and int(idx) < len(c):
                nxt.append(c[int(idx)])
        cur = nxt
    if fanned:
        return cur
    return cur[0] if cur else None


def _dig(obj: Any, dotted: str) -> Any:  # kept for callers that imported it
    return dig(obj, dotted)


def _xml_rows(root: Element, path: str) -> list[dict]:
    """XML: `path` is an element path ("channel/item", ".//entry"); each element becomes a
    row of its attributes and its children's text (namespaces dropped)."""

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    elems = root.findall(path) if path else [root]
    if not elems and path:  # a namespaced document: match on local names
        want = [p for p in path.replace(".//", "").split("/") if p]
        elems = [e for e in root.iter() if local(e.tag) == want[-1]]
    rows = []
    for e in elems:
        row: dict[str, Any] = {local(k): v for k, v in e.attrib.items()}
        for c in e:
            row.setdefault(local(c.tag), (c.text or "").strip())
            for k, v in c.attrib.items():
                row.setdefault(f"{local(c.tag)}.{local(k)}", v)
        if (e.text or "").strip() and "text" not in row:
            row["text"] = e.text.strip()
        rows.append(row)
    return rows


def parse(op: Operation, raw: bytes) -> tuple[list, Any]:
    """(rows, document) by the operation's response format."""
    fmt = op.render.format
    text = raw.decode("utf-8", errors="replace")
    if fmt == "json":
        try:
            doc = json.loads(text) if text.strip() else None
        except ValueError as exc:
            raise ModuleError("the response is not JSON") from exc
        rows = dig(doc, op.render.rows)
        if isinstance(rows, dict):
            rows = [rows]  # a single object is one row
        return (rows if isinstance(rows, list) else ([] if rows is None else [rows])), doc
    if fmt == "xml":
        # defusedxml: a third party's reply must not expand entities or reach out for DTDs.
        try:
            root = ElementTree.fromstring(raw)
        except DefusedXmlException as exc:
            raise ModuleError("the XML response declares entities; refused") from exc
        except ElementTree.ParseError as exc:
            raise ModuleError("the response is not XML") from exc
        return _xml_rows(root, op.render.rows), None
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(text))), None
    return [{"line": ln} for ln in text.splitlines() if ln.strip()], None


# ------------------------------------------------------------------- one call


async def call(
    module: Module,
    cfg: ModuleConfig,
    op: Operation,
    args: dict[str, Any],
    *,
    debug: bool = False,
) -> dict:
    """Run one operation. Returns {ok, rows, count, when, pages, cached}. With `debug` (the
    bay's *try it*), the cache is skipped and the first page's raw text rides along as
    `sample`, so the person sees what came back beside how it rendered."""
    args = coerce(op, args)
    # Keyed on what the call actually is, so an edited operation never answers from a
    # cache filled by its previous shape.
    key = (
        module.id,
        op.id,
        cfg.base_url or module.base_url,
        op.method,
        op.path,
        op.render.format,
        op.render.rows,
        op.tool,
        op.resource,
        tuple(sorted((k, str(v)) for k, v in op.const_query.items())),
        tuple(sorted((k, str(v)) for k, v in args.items())),
    )
    ttl = module.limits.cache_seconds if module.limits else CACHE_TTL
    hit = _cache.get(key)
    now = time.time()
    if hit and now - hit[0] < ttl and not debug:
        return {
            "ok": True,
            "rows": hit[1],
            "count": hit[2],
            "when": hit[0],
            "cached": True,
            "pages": 0,
        }

    if module.is_mcp:
        # An MCP server: one tool call (read-only by the manifest and by the server's
        # word) or one resource read, over the module's live session.
        from library_agent.modules.mcp_transport import MCPError, call_operation

        _rate(module)
        try:
            rows, raw = await call_operation(module, cfg, op, args)
        except MCPError as exc:
            raise ModuleError(str(exc)) from exc
        rows = rows[: max(op.render.limit * 5, op.render.limit)]
        if not debug:
            _cache[key] = (now, rows, len(rows))
        out = {
            "ok": True,
            "rows": rows,
            "count": len(rows),
            "when": now,
            "cached": False,
            "pages": 1,
        }
        if debug:
            out["sample"] = raw[:20_000]
        return out

    base = (cfg.base_url or module.base_url).rstrip("/")
    host = urlparse(base).hostname
    if not host or urlparse(base).scheme not in ("http", "https"):
        raise ModuleError(f"{module.name}: no base URL configured")
    allowed = _hosts(module, base)
    path, query, body = build(op, args)
    p = op.pagination
    pages = max(1, min(p.max_pages, HARD_MAX_PAGES)) if p.type != "none" else 1
    rows: list = []
    fetched = 0
    sample = ""
    async with httpx.AsyncClient(
        base_url=base, timeout=module.limits.timeout, follow_redirects=False
    ) as client:
        auth_headers, auth_query = await _auth(client, module, cfg, base)
        headers = {**module.headers, **cfg.extra_headers, **auth_headers}
        url: str | None = path
        cursor: Any = None
        for i in range(pages):
            q = {**query, **auth_query}
            if p.type in ("page", "offset", "cursor") and p.size_param and p.size:
                q[p.size_param] = str(p.size)
            if p.type == "page":
                q[p.param] = str(p.start + i)
            elif p.type == "offset":
                q[p.param] = str(p.start + i * (p.size or len(rows) or 1))
            elif p.type == "cursor" and cursor not in (None, ""):
                q[p.param] = str(cursor)
            target = url or path
            if (urlparse(target).hostname or host) not in allowed:
                raise ModuleError(f"{module.name}: refused a page off the pinned host")
            _rate(module)
            kw: dict[str, Any] = {
                "params": q if i == 0 or p.type != "link" else None,
                "headers": headers,
            }
            if op.method == "POST":
                kw["json"] = body
            resp, raw = await _fetch(client, module, op.method, target, **kw)
            fetched += 1
            if debug and not sample:
                sample = raw[:20_000].decode("utf-8", errors="replace")
            page_rows, doc = parse(op, raw)
            rows.extend(page_rows)
            if p.type == "none" or len(rows) >= op.render.limit:
                break
            if p.type in ("page", "offset") and (
                not page_rows or (p.size and len(page_rows) < p.size)
            ):
                break
            if p.type == "cursor":
                cursor = dig(doc, p.cursor_path)
                if cursor in (None, "", []):
                    break
            if p.type == "link":
                nxt = resp.links.get("next", {}).get("url")
                if not nxt:
                    break
                url = urljoin(str(resp.request.url), nxt)
    if not debug:
        _cache[key] = (now, rows, len(rows))
    out = {"ok": True, "rows": rows, "count": len(rows), "when": now, "cached": False}
    out["pages"] = fetched
    if debug:
        out["sample"] = sample
    return out
