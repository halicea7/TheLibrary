"""What a module is: a line to an API, described so the librarian can consult it during a
question without ever writing a URL or code.

A module is not a shelf of documents; it is a small, curated set of read-only operations.
Each operation declares the parameters it takes (typed; the picker fills them), when it
applies (a sentence for the planner), the request it maps to (a path template, query and
optionally a JSON body for a read-only search endpoint), how to page through results, and
how what comes back becomes passage text. Everything the model touches is a declared
field; the transport is built from the manifest, not from anything the model emits.

A module can be code (the built-ins) or data: a JSON manifest in
`~/.library-agent/connectors/`, parsed and checked here by `from_dict`. The operator's
secrets are never in a manifest -- they are held beside the provider keys, 0600."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlparse

FORMAT = "library-connector/1"

AUTH_TYPES = ("none", "bearer", "header", "query", "basic", "oauth2_client")
TRANSPORTS = ("http", "mcp_stdio", "mcp_http")
PARAM_TYPES = ("string", "int", "number", "bool", "enum", "date")
PAGINATION = ("none", "page", "offset", "cursor", "link")
RESPONSE_FORMATS = ("json", "xml", "csv", "text")

# Ceilings no manifest can raise: one question must not crawl or flood.
HARD_MAX_PAGES = 10
HARD_MAX_BYTES = 5_000_000
HARD_TIMEOUT = 60.0


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Render:
    """How a response becomes passage text: where the rows are, the line each prints as
    (fields by name, nested with dots, with filters: `{created_at|date}`), what to say
    when there are none, and how many to print. No model call -- a citation must trace to
    what came back."""

    rows: str  # path to the rows: "data", "data.items[*]", "" for the whole response
    line: str  # e.g. "{title} by {user.login} ({created_at|date})"
    empty: str = "nothing matched"
    limit: int = 20
    format: str = "json"  # json | xml | csv | text
    # A line computed over every row fetched, before the rows: "{count} threats on
    # {distinct:agentRealtimeInfo.agentComputerName} endpoints". Counted in code, so a
    # total the answer gives can be checked -- not summed by the model from 30 lines.
    summary: str = ""


@dataclass(frozen=True)
class Pagination:
    type: str = "none"  # none | page | offset | cursor | link
    param: str = ""  # the query key for the page number, offset or cursor
    size_param: str = ""  # the query key for the page size, if any
    size: int = 0  # page size to ask for (and, for offset, the step)
    start: int = 1  # the first page number (page) or offset (offset)
    cursor_path: str = ""  # cursor: where the next cursor is in the response
    max_pages: int = 3


@dataclass(frozen=True)
class Operation:
    id: str
    summary: str  # what it answers, one line
    ask_when: str  # for the picker: when this operation applies
    path: str  # a path template, e.g. "/repos/{owner}/{repo}/issues"
    params: dict[str, Any]  # JSON schema the picker fills (properties / required)
    query: dict[str, str]  # declared-parameter -> query key
    render: Render
    const_query: dict[str, str] = field(default_factory=dict)  # fixed query keys
    method: str = "GET"  # GET, or POST only for a read-only search endpoint
    body: Any = None  # POST: a JSON template; "{param}" strings are replaced by values
    pagination: Pagination = field(default_factory=Pagination)
    # Per parameter: where it goes ("path", "query", "body", "args") and how it is checked.
    param_specs: dict[str, dict] = field(default_factory=dict)
    # An MCP token's operation is a tool, called with the parameters as its arguments; its
    # author must declare it read-only, and a server that marks it otherwise is refused.
    tool: str = ""
    read_only: bool = False
    # Clearance for this operation alone: "local" (local models only), "any", or "" to
    # take the token's. Lets one token keep a sensitive lookup local and share a plain one.
    clearance: str = ""

    def path_params(self) -> list[str]:
        return re.findall(r"\{(\w+)\}", self.path)


@dataclass(frozen=True)
class Auth:
    """How the secret is attached. The secret itself is the operator's, never here."""

    type: str = "header"  # none | bearer | header | query | basic | oauth2_client
    header: str = "Authorization"  # bearer / header
    prefix: str = ""  # header: e.g. "ApiToken " ; bearer implies "Bearer "
    param: str = ""  # query: the parameter the key goes in
    token_url: str = ""  # oauth2_client: where to exchange the client credentials
    scope: str = ""  # oauth2_client: optional scope


@dataclass(frozen=True)
class Transport:
    """How the token reaches its source. "http": the base URL, as everywhere above.
    "mcp_stdio": an MCP server this machine starts (`command` + `args`), its secret handed
    in the environment variable `secret_env`; nothing else of this machine's environment
    but PATH, HOME and the locale goes with it. "mcp_http": an MCP server at `url`."""

    type: str = "http"
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    secret_env: str = ""
    url: str = ""


@dataclass(frozen=True)
class Limits:
    timeout: float = 30.0
    max_bytes: int = 2_000_000
    rate_per_minute: int = 30
    cache_seconds: float = 90.0


@dataclass(frozen=True)
class Module:
    id: str
    name: str
    kind: str  # a short noun for the object/label, e.g. "endpoint security"
    colour: str  # the live-result pigment on this module's citations
    auth_scheme: str  # legacy: "apitoken" or "bearer" (see `auth`)
    auth_header: str  # legacy: the header the token goes in
    operations: tuple[Operation, ...]
    local_only: bool = True  # refuse to consult when chat is on a remote provider
    description: str = ""
    auth: Auth | None = None
    headers: dict[str, str] = field(default_factory=dict)  # fixed request headers
    limits: Limits = field(default_factory=Limits)
    allowed_hosts: tuple[str, ...] = ()  # besides the base URL's own host
    base_url: str = ""  # a suggested base URL; the operator's setting wins
    builtin: bool = False
    source: str = ""  # where a data manifest was loaded from
    transport: Transport = field(default_factory=Transport)

    @property
    def is_mcp(self) -> bool:
        return self.transport.type != "http"

    def op(self, op_id: str) -> Operation | None:
        return next((o for o in self.operations if o.id == op_id), None)

    def op_local_only(self, op: Operation) -> bool:
        """Whether this operation may only run while chat is on a local model."""
        return op.clearance == "local" if op.clearance else self.local_only

    def cleared(self, remote_chat: bool) -> tuple[Operation, ...]:
        """The operations that may run now: all of them on a local model; on a remote
        provider, only those cleared for any model."""
        return tuple(o for o in self.operations if not (remote_chat and self.op_local_only(o)))

    def token_prefix(self) -> str:
        return {"apitoken": "ApiToken ", "bearer": "Bearer "}.get(self.auth_scheme, "")

    def auth_spec(self) -> Auth:
        """The effective auth, for built-ins written before `auth` existed."""
        if self.auth:
            return self.auth
        return Auth(type="header", header=self.auth_header, prefix=self.token_prefix())


# ------------------------------------------------------------------ data manifests

_ID = re.compile(r"^[a-z][a-z0-9_-]{1,39}$")
_KEY = re.compile(r"^[A-Za-z0-9_.\-\[\]]{1,80}$")
_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")
_HEADER = re.compile(r"^[A-Za-z0-9-]{1,60}$")
# Headers a manifest may not set: they belong to the transport or to the secret.
_RESERVED_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "cookie"}


def _s(d: dict, k: str, default: str = "", n: int = 400) -> str:
    v = d.get(k, default)
    if v is None:
        return default
    if not isinstance(v, str):
        raise ManifestError(f"{k} must be text")
    return v.strip()[:n]


def _int(d: dict, k: str, default: int, lo: int, hi: int) -> int:
    v = d.get(k, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ManifestError(f"{k} must be a number")
    return int(min(hi, max(lo, v)))


def _host_of(url: str) -> str:
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ManifestError(f"not an http(s) URL: {url!r}")
    return u.hostname


def _param(name: str, spec: dict, path_names: set[str]) -> tuple[dict, dict]:
    """One declared parameter -> (its JSON schema, its placement spec)."""
    if not isinstance(spec, dict):
        raise ManifestError(f"parameter {name}: a parameter is an object")
    t = spec.get("type", "string")
    if t not in PARAM_TYPES:
        raise ManifestError(f"parameter {name}: type must be one of {', '.join(PARAM_TYPES)}")
    where = spec.get("in") or ("path" if name in path_names else "query")
    if where not in ("path", "query", "body"):
        raise ManifestError(f"parameter {name}: 'in' is path, query or body")
    schema: dict[str, Any] = {"description": _s(spec, "description", n=300)}
    if t == "enum":
        values = spec.get("values")
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(v, str) for v in values)
        ):
            raise ManifestError(f"parameter {name}: an enum lists its values")
        schema.update(type="string", enum=values[:50])
    elif t == "int":
        schema["type"] = "integer"
    elif t == "number":
        schema["type"] = "number"
    elif t == "bool":
        schema["type"] = "boolean"
    elif t == "date":
        schema.update(type="string", description=(schema["description"] + " (YYYY-MM-DD)").strip())
    else:
        schema.update(type="string", maxLength=_int(spec, "max_length", 200, 1, 2000))
        pattern = spec.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str) or len(pattern) > 200:
                raise ManifestError(f"parameter {name}: a pattern is a regular expression")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ManifestError(f"parameter {name}: pattern does not compile ({exc})") from exc
            schema["pattern"] = pattern
    for bound in ("min", "max"):
        if bound in spec and t in ("int", "number"):
            schema["minimum" if bound == "min" else "maximum"] = spec[bound]
    placement = {
        "type": t,
        "in": where,
        "key": _s(spec, "key", name, 80) or name,
        "required": bool(spec.get("required")) or where == "path",
        "default": spec.get("default"),
        "min": spec.get("min"),
        "max": spec.get("max"),
        "values": schema.get("enum"),
        "max_length": schema.get("maxLength"),
        # Must match the source's own listing exactly (a vendor as the inventory spells
        # it): never filled from a reader's question, where "Google" for "Google LLC"
        # would find nothing and read as "not installed".
        "exact": bool(spec.get("exact")),
        # What a value must look like (a CVE id: CVE-\d{4}-\d+). A value from the question
        # that doesn't match is dropped -- "BERT" is not a CVE, and a guess is not a lookup.
        "pattern": schema.get("pattern"),
    }
    return schema, placement


def _mcp_operation(o: dict, oid: str) -> Operation:
    tool = _s(o, "tool", n=100)
    if not re.match(r"^[A-Za-z0-9_.\-/]{1,100}$", tool):
        raise ManifestError(f"{oid}: an MCP operation names its tool")
    if o.get("read_only") is not True:
        raise ManifestError(
            f"{oid}: mark it read_only: true -- only tools that look things up, never ones "
            "that change anything"
        )
    raw_params = o.get("params") or {}
    if not isinstance(raw_params, dict):
        raise ManifestError(f"{oid}: params is an object of name -> spec")
    props, placements, required = {}, {}, []
    for name, spec in raw_params.items():
        if not re.match(r"^[a-zA-Z_][\w\-]{0,63}$", name):
            raise ManifestError(f"{oid}: parameter name {name!r}")
        schema, place = _param(
            name, {**spec, "in": "query"} if isinstance(spec, dict) else spec, set()
        )
        place["in"] = "args"
        props[name], placements[name] = schema, place
        if place["required"]:
            required.append(name)
    const = o.get("args") or {}
    if not isinstance(const, dict):
        raise ManifestError(f"{oid}: args is fixed name -> value")
    r = o.get("response") or {}
    fmt = _s(r, "format", "json", 8)
    if fmt not in ("json", "text"):
        raise ManifestError(f"{oid}: an MCP tool's result is read as json or text")
    return Operation(
        id=oid,
        summary=_s(o, "summary", n=300),
        ask_when=_s(o, "ask_when", n=400),
        path="",
        params={"type": "object", "properties": props, "required": required},
        query={},
        const_query={str(k): v for k, v in const.items()},
        render=Render(
            rows=_s(r, "rows", "", 200),
            line=_s(r, "line", "{text}", 400) or "{text}",
            empty=_s(r, "empty", "nothing matched", 300),
            limit=_int(r, "limit", 20, 1, 200),
            format=fmt,
            summary=_s(r, "summary", "", 300),
        ),
        method="MCP",
        param_specs=placements,
        tool=tool,
        read_only=True,
    )


def _operation(o: dict, seen: set[str], *, mcp: bool = False) -> Operation:
    if not isinstance(o, dict):
        raise ManifestError("an operation is an object")
    oid = _s(o, "id", n=40)
    if not _ID.match(oid) or oid in seen:
        raise ManifestError(f"operation id {oid!r} must be unique, lowercase, 2-40 characters")
    seen.add(oid)
    clearance = o.get("clearance") or ""
    if clearance not in ("", "local", "any"):
        raise ManifestError(f"{oid}: clearance is local, any, or left out to take the token's")
    op = _mcp_operation(o, oid) if mcp else _http_operation(o, oid)
    return replace(op, clearance=clearance) if clearance else op


def _http_operation(o: dict, oid: str) -> Operation:
    method = _s(o, "method", "GET", 8).upper()
    if method not in ("GET", "POST"):
        raise ManifestError(f"{oid}: only GET, or POST for a read-only search")
    if method == "POST" and o.get("read_only_post") is not True:
        raise ManifestError(
            f"{oid}: a POST must be marked read_only_post: true -- for search endpoints that "
            "take a query body, never for anything that changes data"
        )
    path = _s(o, "path", n=400)
    if not path.startswith("/") or "://" in path or ".." in path:
        raise ManifestError(f"{oid}: path must be absolute on the base URL (start with /)")
    path_names = set(re.findall(r"\{(\w+)\}", path))
    raw_params = o.get("params") or {}
    if not isinstance(raw_params, dict):
        raise ManifestError(f"{oid}: params is an object of name -> spec")
    props, placements, required = {}, {}, []
    for name, spec in raw_params.items():
        if not re.match(r"^[a-zA-Z_]\w{0,39}$", name):
            raise ManifestError(f"{oid}: parameter name {name!r}")
        schema, place = _param(name, spec, path_names)
        props[name], placements[name] = schema, place
        if place["required"]:
            required.append(name)
    missing = path_names - set(props)
    if missing:
        raise ManifestError(f"{oid}: path uses undeclared {', '.join(sorted(missing))}")
    const = o.get("query") or {}
    if not isinstance(const, dict) or not all(
        isinstance(v, (str, int, float)) for v in const.values()
    ):
        raise ManifestError(f"{oid}: query is fixed key -> value")
    r = o.get("response") or {}
    fmt = _s(r, "format", "json", 8)
    if fmt not in RESPONSE_FORMATS:
        raise ManifestError(f"{oid}: response format is one of {', '.join(RESPONSE_FORMATS)}")
    line = _s(r, "line", n=400)
    if not line:
        raise ManifestError(f"{oid}: response.line says how a row prints")
    render = Render(
        rows=_s(r, "rows", "", 200),
        line=line,
        empty=_s(r, "empty", "nothing matched", 300),
        limit=_int(r, "limit", 20, 1, 200),
        format=fmt,
        summary=_s(r, "summary", "", 300),
    )
    p = o.get("pagination") or {}
    ptype = _s(p, "type", "none", 10)
    if ptype not in PAGINATION:
        raise ManifestError(f"{oid}: pagination is one of {', '.join(PAGINATION)}")
    if ptype in ("page", "offset", "cursor") and not _s(p, "param", n=80):
        raise ManifestError(f"{oid}: {ptype} pagination names its query parameter")
    if ptype == "cursor" and not _s(p, "cursor_path", n=200):
        raise ManifestError(f"{oid}: cursor pagination says where the next cursor is")
    pagination = Pagination(
        type=ptype,
        param=_s(p, "param", n=80),
        size_param=_s(p, "size_param", n=80),
        size=_int(p, "size", 0, 0, 1000),
        start=_int(p, "start", 1 if ptype == "page" else 0, 0, 10_000),
        cursor_path=_s(p, "cursor_path", n=200),
        max_pages=_int(p, "max_pages", 3, 1, HARD_MAX_PAGES),
    )
    return Operation(
        id=oid,
        summary=_s(o, "summary", n=300),
        ask_when=_s(o, "ask_when", n=400),
        path=path,
        params={"type": "object", "properties": props, "required": required},
        query={n: pl["key"] for n, pl in placements.items() if pl["in"] == "query"},
        const_query={str(k): str(v) for k, v in const.items()},
        render=render,
        method=method,
        body=o.get("body") if method == "POST" else None,
        pagination=pagination,
        param_specs=placements,
    )


def from_dict(d: dict, *, source: str = "", builtin: bool = False) -> Module:
    """A data manifest, checked field by field. Raises ManifestError naming what is wrong."""
    if not isinstance(d, dict):
        raise ManifestError("a connector is a JSON object")
    if d.get("format") not in (None, FORMAT):
        raise ManifestError(f"unknown format {d.get('format')!r}; this library reads {FORMAT}")
    mid = _s(d, "id", n=40)
    if not _ID.match(mid):
        raise ManifestError("id must be lowercase letters, digits, - or _, 2-40 characters")
    name = _s(d, "name", n=60)
    if not name:
        raise ManifestError("a connector has a name")
    colour = _s(d, "colour", "#4f9186", 7) or "#4f9186"
    if not _COLOUR.match(colour):
        raise ManifestError("colour is #rrggbb")
    a = d.get("auth") or {"type": "none"}
    if not isinstance(a, dict) or a.get("type") not in AUTH_TYPES:
        raise ManifestError(f"auth.type is one of {', '.join(AUTH_TYPES)}")
    auth = Auth(
        type=a["type"],
        header=_s(a, "header", "Authorization", 60) or "Authorization",
        # Verbatim: the trailing space in "ApiToken " is part of the scheme.
        prefix=(a.get("prefix") or "")[:40]
        if a["type"] == "header" and isinstance(a.get("prefix"), str)
        else "",
        param=_s(a, "param", "", 80),
        token_url=_s(a, "token_url", "", 400),
        scope=_s(a, "scope", "", 200),
    )
    if auth.type in ("bearer", "header") and not _HEADER.match(auth.header):
        raise ManifestError("auth.header is a header name")
    if auth.type == "query" and not auth.param:
        raise ManifestError("auth.param names the query parameter the key goes in")
    if auth.type == "oauth2_client" and not auth.token_url:
        raise ManifestError("auth.token_url is where client credentials are exchanged")
    headers = d.get("headers") or {}
    if not isinstance(headers, dict):
        raise ManifestError("headers is name -> value")
    for h, v in headers.items():
        if not _HEADER.match(str(h)) or h.lower() in _RESERVED_HEADERS or not isinstance(v, str):
            raise ManifestError(f"header {h!r} cannot be set by a connector")
        if h.lower() == auth.header.lower() and auth.type in ("bearer", "header"):
            raise ManifestError(f"header {h!r} carries the secret; it is set by auth")
    lim = d.get("limits") or {}
    if not isinstance(lim, dict):
        raise ManifestError("limits is an object")
    limits = Limits(
        timeout=float(min(HARD_TIMEOUT, max(1.0, lim.get("timeout", 30)))),
        max_bytes=_int(lim, "max_bytes", 2_000_000, 10_000, HARD_MAX_BYTES),
        rate_per_minute=_int(lim, "rate_per_minute", 30, 1, 600),
        cache_seconds=float(min(3600, max(0, lim.get("cache_seconds", 90)))),
    )
    hosts = d.get("allowed_hosts") or []
    if not isinstance(hosts, list) or not all(
        isinstance(h, str) and re.match(r"^[a-z0-9.-]+$", h) for h in hosts
    ):
        raise ManifestError("allowed_hosts lists host names")
    base = _s(d, "base_url", "", 400)
    if base:
        _host_of(base)
    if auth.type == "oauth2_client" and "://" in auth.token_url:
        th = _host_of(auth.token_url)
        if base and th != _host_of(base) and th not in hosts:
            raise ManifestError("auth.token_url's host must be the base URL's or in allowed_hosts")
    t = d.get("transport") or {"type": "http"}
    if not isinstance(t, dict) or t.get("type", "http") not in TRANSPORTS:
        raise ManifestError(f"transport.type is one of {', '.join(TRANSPORTS)}")
    ttype = t.get("type", "http")
    targs = t.get("args") or []
    if not isinstance(targs, list) or not all(isinstance(a, str) and len(a) < 500 for a in targs):
        raise ManifestError("transport.args is a list of strings")
    tenv = t.get("env") or {}
    if not isinstance(tenv, dict) or not all(
        re.match(r"^[A-Z_][A-Z0-9_]{0,63}$", str(k)) and isinstance(v, str) for k, v in tenv.items()
    ):
        raise ManifestError("transport.env is NAME -> value")
    transport = Transport(
        type=ttype,
        command=_s(t, "command", n=300),
        args=tuple(targs[:40]),
        env={str(k): v for k, v in tenv.items()},
        secret_env=_s(t, "secret_env", n=64),
        url=_s(t, "url", n=400),
    )
    if ttype == "mcp_stdio":
        # A program, not a shell line: no pipes, no substitutions, arguments given apart.
        if not re.match(r"^[A-Za-z0-9_./\-]{1,300}$", transport.command):
            raise ManifestError("transport.command is a program name or path, without a shell")
        if transport.secret_env and not re.match(r"^[A-Z_][A-Z0-9_]{0,63}$", transport.secret_env):
            raise ManifestError("transport.secret_env is an environment variable name")
    if ttype == "mcp_http":
        _host_of(transport.url or "x")
    ops = d.get("operations") or []
    if not isinstance(ops, list) or not ops:
        raise ManifestError("a connector has at least one operation")
    if len(ops) > 25:
        raise ManifestError("at most 25 operations: a connector is a curated set, not a whole API")
    seen: set[str] = set()
    operations = tuple(_operation(o, seen, mcp=ttype != "http") for o in ops)
    return Module(
        id=mid,
        name=name,
        kind=_s(d, "kind", "live source", 40) or "live source",
        colour=colour.lower(),
        auth_scheme="",
        auth_header=auth.header,
        operations=operations,
        local_only=d.get("clearance", "local") != "any",
        description=_s(d, "description", n=600),
        auth=auth,
        headers={str(k): str(v) for k, v in headers.items()},
        limits=limits,
        allowed_hosts=tuple(h.lower() for h in hosts),
        base_url=base.rstrip("/"),
        builtin=builtin,
        source=source,
        transport=transport,
    )


def to_dict(m: Module) -> dict:
    """A module back to its manifest -- for sharing (never carries a secret) and editing."""
    a = m.auth_spec()
    ops = []
    for o in m.operations:
        params = {}
        for name, schema in (o.params.get("properties") or {}).items():
            pl = o.param_specs.get(name) or {}
            t = pl.get("type") or {"integer": "int", "number": "number", "boolean": "bool"}.get(
                schema.get("type"), "enum" if schema.get("enum") else "string"
            )
            spec = {"type": t, "description": schema.get("description", "")}
            where = pl.get("in") or ("path" if name in o.path_params() else "query")
            spec["in"] = where
            key = pl.get("key") or o.query.get(name) or name
            if key != name:
                spec["key"] = key
            if name in (o.params.get("required") or []):
                spec["required"] = True
            if schema.get("enum"):
                spec["values"] = schema["enum"]
            for k in ("default", "min", "max"):
                if pl.get(k) is not None:
                    spec[k] = pl[k]
            if pl.get("exact"):
                spec["exact"] = True
            if pl.get("pattern"):
                spec["pattern"] = pl["pattern"]
            params[name] = spec
        if o.method == "MCP":
            for spec in params.values():
                spec.pop("in", None)
                spec.pop("key", None)
            ops.append(
                {
                    "id": o.id,
                    "summary": o.summary,
                    "ask_when": o.ask_when,
                    "tool": o.tool,
                    "read_only": True,
                    **({"clearance": o.clearance} if o.clearance else {}),
                    "params": params,
                    "args": dict(o.const_query),
                    "response": {
                        "format": o.render.format,
                        "rows": o.render.rows,
                        "line": o.render.line,
                        "empty": o.render.empty,
                        "limit": o.render.limit,
                        **({"summary": o.render.summary} if o.render.summary else {}),
                    },
                }
            )
            continue
        op: dict[str, Any] = {
            "id": o.id,
            "summary": o.summary,
            "ask_when": o.ask_when,
            "method": o.method,
            "path": o.path,
            "params": params,
            "query": dict(o.const_query),
            "response": {
                "format": o.render.format,
                "rows": o.render.rows,
                "line": o.render.line,
                "empty": o.render.empty,
                "limit": o.render.limit,
                **({"summary": o.render.summary} if o.render.summary else {}),
            },
        }
        if o.clearance:
            op["clearance"] = o.clearance
        if o.method == "POST":
            op["read_only_post"] = True
            op["body"] = o.body
        if o.pagination.type != "none":
            p = o.pagination
            op["pagination"] = {
                k: v
                for k, v in {
                    "type": p.type,
                    "param": p.param,
                    "size_param": p.size_param,
                    "size": p.size,
                    "start": p.start,
                    "cursor_path": p.cursor_path,
                    "max_pages": p.max_pages,
                }.items()
                if v not in ("", None)
            }
        ops.append(op)
    auth: dict[str, Any] = {"type": a.type}
    if a.type in ("bearer", "header"):
        auth["header"] = a.header
    if a.type == "header" and a.prefix:
        auth["prefix"] = a.prefix
    if a.type == "query":
        auth["param"] = a.param
    if a.type == "oauth2_client":
        auth.update(token_url=a.token_url, scope=a.scope)
    return {
        "format": FORMAT,
        "id": m.id,
        "name": m.name,
        "kind": m.kind,
        "colour": m.colour,
        "description": m.description,
        "base_url": m.base_url,
        "clearance": "local" if m.local_only else "any",
        "auth": auth,
        "headers": dict(m.headers),
        "limits": {
            "timeout": m.limits.timeout,
            "max_bytes": m.limits.max_bytes,
            "rate_per_minute": m.limits.rate_per_minute,
            "cache_seconds": m.limits.cache_seconds,
        },
        "allowed_hosts": list(m.allowed_hosts),
        **(
            {
                "transport": {
                    k: v
                    for k, v in {
                        "type": m.transport.type,
                        "command": m.transport.command,
                        "args": list(m.transport.args),
                        "env": dict(m.transport.env),
                        "secret_env": m.transport.secret_env,
                        "url": m.transport.url,
                    }.items()
                    if v not in ("", [], {})
                }
            }
            if m.is_mcp
            else {}
        ),
        "operations": ops,
    }


# Headers whose values are about the protocol, not about you -- kept when a module is
# shared. Any other header's value is yours (an organisation id, a tenant, a key) and is
# left for the person importing it to set.
SHAREABLE_HEADERS = {"accept", "content-type", "accept-language", "user-agent"}


def for_sharing(m: Module) -> tuple[dict, dict]:
    """A module as others may have it: its overall configuration -- what it asks, how it
    reads the answers, how it signs in -- without anything of this installation: the base
    URL or server URL, header and environment values, allowed hosts, an absolute token URL,
    paths under this home directory. Returns (manifest, requires) where `requires` names
    what the importer must fill in. Secrets are never in a manifest to begin with."""
    import os

    d = to_dict(m)
    requires: dict = {}
    if d.get("base_url"):
        d["base_url"] = ""
        requires["base_url"] = True
    if d.get("allowed_hosts"):
        requires["allowed_hosts"] = len(d["allowed_hosts"])
        d["allowed_hosts"] = []
    kept, needed = {}, []
    for k, v in (d.get("headers") or {}).items():
        if k.lower() in SHAREABLE_HEADERS:
            kept[k] = v
        else:
            needed.append(k)
    d["headers"] = kept
    if needed:
        requires["headers"] = needed
    a = d.get("auth") or {}
    if a.get("token_url", "").startswith(("http://", "https://")):
        from urllib.parse import urlparse

        a["token_url"] = urlparse(a["token_url"]).path or "/"
        requires["token_url_host"] = True
    t = d.get("transport")
    if t:
        home = os.path.expanduser("~")
        scrub = lambda s: s.replace(home, "~") if isinstance(s, str) else s
        t["command"] = scrub(t.get("command", ""))
        t["args"] = [scrub(x) for x in t.get("args") or []]
        if t.get("env"):
            requires["env"] = sorted(t["env"])
            t["env"] = {k: "" for k in t["env"]}
        if t.get("url"):
            t["url"] = ""
            requires["server_url"] = True
    d.pop("builtin", None)
    d.pop("source", None)
    return d, requires
