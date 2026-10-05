"""Drafting a connector instead of typing one: from an OpenAPI / Swagger document, or from
a curl command that already works.

Both produce a *draft* manifest -- plain data the person reviews, tries and saves -- never
a live connector. A secret found in a curl command is handed back separately to fill the
connection form; it is never written into the manifest."""

from __future__ import annotations

import json
import re
import shlex
from typing import Any
from urllib.parse import parse_qsl, urlparse

import yaml

from library_agent.modules.manifest import FORMAT, ManifestError


def _slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return (s or "op")[:n]


# --------------------------------------------------------------------- OpenAPI


def load_spec(text: str) -> dict:
    try:
        spec = json.loads(text)
    except ValueError:
        try:
            spec = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ManifestError(f"not JSON or YAML: {exc}") from exc
    if not isinstance(spec, dict) or not ("openapi" in spec or "swagger" in spec):
        raise ManifestError("not an OpenAPI or Swagger document")
    return spec


def _ref(spec: dict, obj: Any, depth: int = 0) -> Any:
    """Follow a local $ref ("#/components/schemas/X"), a few levels deep."""
    while isinstance(obj, dict) and "$ref" in obj and depth < 8:
        path = obj["$ref"]
        if not path.startswith("#/"):
            return {}
        cur: Any = spec
        for part in path[2:].split("/"):
            cur = (
                cur.get(part.replace("~1", "/").replace("~0", "~"), {})
                if isinstance(cur, dict)
                else {}
            )
        obj, depth = cur, depth + 1
    return obj


def _base_url(spec: dict) -> str:
    if spec.get("servers"):
        return str(spec["servers"][0].get("url", "")).rstrip("/")
    host = spec.get("host")
    if host:
        scheme = (spec.get("schemes") or ["https"])[0]
        return f"{scheme}://{host}{spec.get('basePath', '')}".rstrip("/")
    return ""


def _auth(spec: dict) -> dict:
    schemes = (
        (spec.get("components") or {}).get("securitySchemes")
        or spec.get("securityDefinitions")
        or {}
    )
    for s in schemes.values():
        s = _ref(spec, s)
        t, where = s.get("type"), s.get("in")
        if t == "http" and s.get("scheme", "").lower() == "bearer":
            return {"type": "bearer"}
        if t == "http" and s.get("scheme", "").lower() == "basic" or t == "basic":
            return {"type": "basic"}
        if t == "apiKey" and where == "header":
            return {"type": "header", "header": s.get("name", "Authorization")}
        if t == "apiKey" and where == "query":
            return {"type": "query", "param": s.get("name", "api_key")}
        if t == "oauth2":
            flows = s.get("flows") or {}
            cc = flows.get("clientCredentials") or (
                {"tokenUrl": s.get("tokenUrl")} if s.get("flow") == "application" else None
            )
            if cc and cc.get("tokenUrl"):
                return {"type": "oauth2_client", "token_url": cc["tokenUrl"]}
    return {"type": "none"}


def _type(schema: dict) -> tuple[str, list | None]:
    if schema.get("enum"):
        return "enum", [str(v) for v in schema["enum"]][:50]
    t = schema.get("type")
    if t == "integer":
        return "int", None
    if t == "number":
        return "number", None
    if t == "boolean":
        return "bool", None
    if schema.get("format") in ("date", "date-time"):
        return "date", None
    return "string", None


def _rows_and_fields(spec: dict, op: dict) -> tuple[str, list[str]]:
    """Guess where the rows are in a 200 response, and a few fields worth printing."""
    ok = (op.get("responses") or {}).get("200") or (op.get("responses") or {}).get("default") or {}
    ok = _ref(spec, ok)
    schema = ok.get("schema")
    if not schema:
        content = ok.get("content") or {}
        media = content.get("application/json") or next(iter(content.values()), {})
        schema = media.get("schema") or {}
    schema = _ref(spec, schema)
    rows, item = "", schema
    if schema.get("type") == "array":
        item = _ref(spec, schema.get("items") or {})
    else:
        for name, prop in (schema.get("properties") or {}).items():
            prop = _ref(spec, prop)
            if prop.get("type") == "array":
                rows, item = name, _ref(spec, prop.get("items") or {})
                break
    fields = []
    for name, prop in (item.get("properties") or {}).items():
        prop = _ref(spec, prop)
        if prop.get("type") in ("string", "integer", "number", "boolean") or prop.get("enum"):
            fields.append(name)
    prefer = [
        f
        for f in fields
        if re.search(r"name|title|summary|id$|status|state|severity|host", f, re.IGNORECASE)
    ]
    return rows, (prefer + [f for f in fields if f not in prefer])[:4]


def openapi_operations(spec: dict) -> list[dict]:
    """The GET operations a document declares, with their parameters -- for picking."""
    out = []
    for path, item in (spec.get("paths") or {}).items():
        item = _ref(spec, item)
        shared = item.get("parameters") or []
        op = item.get("get")
        if not isinstance(op, dict):
            continue
        params = []
        for p in list(shared) + list(op.get("parameters") or []):
            p = _ref(spec, p)
            if p.get("in") not in ("path", "query"):
                continue
            t, values = _type(_ref(spec, p.get("schema") or p))
            params.append(
                {
                    "name": p.get("name"),
                    "in": p["in"],
                    "type": t,
                    "values": values,
                    "required": bool(p.get("required")) or p["in"] == "path",
                    "description": (p.get("description") or "")[:200],
                }
            )
        rows, fields = _rows_and_fields(spec, op)
        out.append(
            {
                "key": f"GET {path}",
                "operation_id": op.get("operationId") or _slug(f"get_{path}"),
                "path": path,
                "summary": (op.get("summary") or op.get("description") or "").strip()[:200],
                "params": params,
                "rows": rows,
                "fields": fields,
            }
        )
    return out


def draft_from_openapi(spec: dict, keys: list[str], *, name: str = "", base_url: str = "") -> dict:
    """A draft manifest from the picked operations ("GET /path" keys)."""
    info = spec.get("info") or {}
    title = name or info.get("title") or "Connector"
    picked = [o for o in openapi_operations(spec) if o["key"] in set(keys)]
    if not picked:
        raise ManifestError("pick at least one operation")
    if len(picked) > 25:
        raise ManifestError("pick at most 25 operations")
    ops, seen = [], set()
    for o in picked:
        oid = _slug(o["operation_id"])
        while oid in seen:
            oid = _slug(oid + "_2")
        seen.add(oid)
        params = {}
        for p in o["params"]:
            pname = re.sub(r"\W", "_", p["name"] or "")[:40] or "param"
            if not re.match(r"^[a-zA-Z_]", pname):
                pname = "p_" + pname
            spec_p: dict[str, Any] = {
                "type": p["type"],
                "in": p["in"],
                "description": p["description"],
            }
            if pname != p["name"]:
                spec_p["key"] = p["name"]
            if p["required"]:
                spec_p["required"] = True
            if p["values"]:
                spec_p["values"] = p["values"]
            params[pname] = spec_p
        path = o["path"]
        for p in o["params"]:
            if p["in"] == "path":
                safe = re.sub(r"\W", "_", p["name"])
                path = path.replace("{" + p["name"] + "}", "{" + safe + "}")
        line = " — ".join("{" + f + "}" for f in o["fields"]) or "{value}"
        ops.append(
            {
                "id": oid,
                "summary": o["summary"] or f"GET {o['path']}",
                "ask_when": f"the question asks for {o['summary'].rstrip('.').lower() or 'this'}.",
                "path": path,
                "params": params,
                "response": {
                    "rows": o["rows"],
                    "line": line,
                    "empty": "nothing matched",
                    "limit": 20,
                },
            }
        )
    return {
        "format": FORMAT,
        "id": _slug(title, 30).replace("_", "-") or "connector",
        "name": title[:60],
        "kind": "live source",
        "colour": "#4f9186",
        "description": (info.get("description") or "")[:600],
        "base_url": base_url or _base_url(spec),
        "clearance": "local",
        "auth": _auth(spec),
        "operations": ops,
    }


# --------------------------------------------------------------------- curl


def draft_from_curl(cmd: str) -> tuple[dict, dict, list[str]]:
    """(draft manifest, credentials found, notes). One operation, from one working command.
    The command's query values become parameters with those values as defaults. A POST is
    drafted without read_only_post: it will not save until the person says it only
    searches."""
    try:
        argv = shlex.split(cmd.replace("\\\n", " "))
    except ValueError as exc:
        raise ManifestError(f"could not read that command: {exc}") from exc
    if not argv or argv[0] != "curl":
        raise ManifestError("paste a curl command")
    method, url, headers, user, data = "GET", "", {}, "", None
    i = 1
    while i < len(argv):
        a = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else ""
        if a in ("-X", "--request"):
            method, i = nxt.upper(), i + 2
        elif a in ("-H", "--header"):
            k, _, v = nxt.partition(":")
            headers[k.strip()] = v.strip()
            i += 2
        elif a in ("-u", "--user"):
            user, i = nxt, i + 2
        elif a in ("-d", "--data", "--data-raw", "--data-binary", "--json"):
            data, i = nxt, i + 2
            if method == "GET":
                method = "POST"
        elif a.startswith(("http://", "https://")):
            url, i = a, i + 1
        elif a in ("--url",):
            url, i = nxt, i + 2
        else:
            i += 1
    if not url:
        raise ManifestError("no URL in that command")
    if method not in ("GET", "POST"):
        raise ManifestError(f"a {method} changes data; connectors only read")
    u = urlparse(url)
    base = f"{u.scheme}://{u.netloc}"
    creds: dict[str, str] = {}
    auth: dict[str, Any] = {"type": "none"}
    fixed_headers: dict[str, str] = {}
    for k, v in headers.items():
        low = k.lower()
        if low == "authorization":
            scheme, _, secret = v.partition(" ")
            if scheme.lower() == "bearer":
                auth = {"type": "bearer"}
            else:
                auth = {"type": "header", "header": k, "prefix": scheme + " " if secret else ""}
                secret = secret or v
            creds["token"] = secret
        elif re.search(r"api[-_]?key|token|secret", low):
            auth = {"type": "header", "header": k}
            creds["token"] = v
        elif low in ("accept", "content-type") or low.startswith("x-"):
            fixed_headers[k] = v
    if user:
        name, _, pw = user.partition(":")
        auth, creds = {"type": "basic"}, {"username": name, "token": pw}
    params: dict[str, Any] = {}
    for k, v in parse_qsl(u.query, keep_blank_values=True):
        if re.search(r"api[-_]?key|token|secret|access_key", k, re.IGNORECASE):
            auth, creds = {"type": "query", "param": k}, {"token": v}
            continue
        pname = re.sub(r"\W", "_", k)[:40] or "param"
        if not re.match(r"^[a-zA-Z_]", pname):
            pname = "p_" + pname
        spec: dict[str, Any] = {
            "type": "int" if v.isdigit() else "string",
            "in": "query",
            "default": int(v) if v.isdigit() else v,
        }
        if pname != k:
            spec["key"] = k
        params[pname] = spec
    op: dict[str, Any] = {
        "id": _slug(u.path.strip("/").split("/")[-1] or "query"),
        "method": method,
        "summary": f"{method} {u.path}",
        "ask_when": "the question asks for what this returns.",
        "path": u.path or "/",
        "params": params,
        "response": {"rows": "", "line": "{value}", "empty": "nothing matched", "limit": 20},
    }
    if method == "POST":
        try:
            op["body"] = json.loads(data) if data else {}
        except ValueError as exc:
            raise ManifestError(
                "the request body is not JSON; only JSON search bodies are read"
            ) from exc
    notes = []
    if method == "POST":
        notes.append(
            "This is a POST. Save it only if the endpoint searches and changes nothing — "
            "then mark it read-only (read_only_post). Until then the draft will not save."
        )
    if creds:
        notes.append(
            "Credentials were found in the command; they go in the connection, not the connector file."
        )
    return (
        {
            "format": FORMAT,
            "id": _slug(u.hostname or "connector", 30).replace("_", "-"),
            "name": (u.hostname or "Connector")[:60],
            "kind": "live source",
            "colour": "#4f9186",
            "base_url": base,
            "clearance": "local",
            "auth": auth,
            "headers": fixed_headers,
            "operations": [op],
        },
        creds,
        notes,
    )


# --------------------------------------------------------------------- Postman

_VAR = re.compile(r"\{\{\s*([\w.\-]+)\s*\}\}")
_SECRET_NAME = re.compile(r"api[-_]?key|token|secret|password|access_key|bearer", re.IGNORECASE)


def load_postman(text: str) -> dict:
    try:
        col = json.loads(text)
    except ValueError as exc:
        raise ManifestError("not a JSON Postman collection") from exc
    info = col.get("info") if isinstance(col, dict) else None
    if not isinstance(info, dict) or not isinstance(col.get("item"), list):
        raise ManifestError("not a Postman collection (v2.x)")
    return col


def _pm_vars(col: dict) -> dict[str, str]:
    return {
        str(v.get("key")): str(v.get("value", ""))
        for v in col.get("variable") or []
        if isinstance(v, dict) and v.get("key")
    }


def _pm_items(items: list, folder: str = "") -> list[tuple[str, dict]]:
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        name = f"{folder}/{it.get('name', '')}".strip("/")
        if isinstance(it.get("item"), list):
            out += _pm_items(it["item"], name)
        elif isinstance(it.get("request"), dict):
            out.append((name, it))
    return out


def _pm_url(req: dict, vars_: dict) -> tuple[str, str, list[str], list[tuple[str, str]]]:
    """(scheme://host, raw path, path segments, query pairs), variables resolved where the
    collection gives them a value."""
    url = req.get("url")
    if isinstance(url, str):
        url = {"raw": url}
    url = url or {}
    sub = lambda s: _VAR.sub(lambda m: vars_.get(m.group(1), m.group(0)), str(s))
    raw = sub(url.get("raw", ""))
    if url.get("host") is not None or url.get("path") is not None:
        host = sub(
            ".".join(url["host"]) if isinstance(url.get("host"), list) else url.get("host", "")
        )
        proto = url.get("protocol") or (urlparse(raw).scheme if "://" in raw else "https")
        path = url.get("path") or []
        segs = [
            sub(p if isinstance(p, str) else p.get("value", ""))
            for p in (path if isinstance(path, list) else str(path).split("/"))
        ]
        query = [
            (str(q.get("key")), sub(q.get("value", "")))
            for q in url.get("query") or []
            if isinstance(q, dict) and not q.get("disabled")
        ]
        base = f"{proto}://{host}" if host and "://" not in host else host
    else:
        u = urlparse(raw)
        base = f"{u.scheme}://{u.netloc}"
        segs = [s for s in u.path.split("/") if s]
        query = parse_qsl(u.query, keep_blank_values=True)
    return base.rstrip("/"), raw, [s for s in segs if s], query


def postman_requests(col: dict) -> list[dict]:
    vars_ = _pm_vars(col)
    out = []
    for name, it in _pm_items(col.get("item")):
        req = it["request"]
        method = str(req.get("method", "GET")).upper()
        base, _raw, segs, _query = _pm_url(req, vars_)
        out.append(
            {
                "key": name,
                "name": name,
                "method": method,
                "url": f"{base}/{'/'.join(segs)}",
                "usable": method in ("GET", "POST"),
                "note": ""
                if method == "GET"
                else "a POST: only if it only searches"
                if method == "POST"
                else "changes data; cannot be a token",
            }
        )
    return out


def _pm_auth(auth: dict | None, vars_: dict) -> tuple[dict, dict]:
    if not isinstance(auth, dict):
        return {"type": "none"}, {}
    kind = auth.get("type")
    fields = {
        str(f.get("key")): str(f.get("value", ""))
        for f in auth.get(kind) or []
        if isinstance(f, dict)
    }
    val = lambda k: _VAR.sub(lambda m: vars_.get(m.group(1), ""), fields.get(k, ""))
    if kind == "bearer":
        return {"type": "bearer"}, {"token": val("token")}
    if kind == "basic":
        return {"type": "basic"}, {"username": val("username"), "token": val("password")}
    if kind == "apikey":
        where, key = fields.get("in", "header"), fields.get("key", "X-API-Key")
        if where == "query":
            return {"type": "query", "param": key}, {"token": val("value")}
        return {"type": "header", "header": key}, {"token": val("value")}
    if kind == "oauth2" and fields.get("grant_type", "").startswith("client"):
        return {"type": "oauth2_client", "token_url": val("accessTokenUrl")}, {
            "username": val("clientId"),
            "token": val("clientSecret"),
        }
    return {"type": "none"}, {}


def draft_from_postman(col: dict, keys: list[str]) -> tuple[dict, dict, list[str]]:
    """(draft manifest, credentials, notes) from the picked requests. `:id` path segments
    and unresolved {{variables}} become parameters; query values become parameters with
    those values as defaults; a secret in a variable goes to the connection."""
    vars_ = _pm_vars(col)
    wanted = set(keys)
    picked = [(n, it) for n, it in _pm_items(col.get("item")) if n in wanted]
    if not picked:
        raise ManifestError("pick at least one request")
    if len(picked) > 25:
        raise ManifestError("pick at most 25 requests")
    auth, creds = _pm_auth(col.get("auth"), vars_)
    bases, ops, notes, seen = set(), [], [], set()
    headers: dict[str, str] = {}
    for name, it in picked:
        req = it["request"]
        method = str(req.get("method", "GET")).upper()
        if method not in ("GET", "POST"):
            raise ManifestError(f"{name}: a {method} changes data; connectors only read")
        if req.get("auth"):
            auth, creds = _pm_auth(req["auth"], vars_)
        base, _raw, segs, query = _pm_url(req, vars_)
        bases.add(base)
        params: dict[str, Any] = {}
        path_parts = []
        for seg in segs:
            m = _VAR.fullmatch(seg) or re.fullmatch(r":(\w+)", seg)
            if m:
                pname = re.sub(r"\W", "_", m.group(1))[:40]
                params[pname] = {
                    "type": "string",
                    "in": "path",
                    "required": True,
                    "description": "",
                }
                path_parts.append("{" + pname + "}")
            else:
                path_parts.append(seg)
        for k, v in query:
            if _SECRET_NAME.search(k):
                auth, creds = (
                    {"type": "query", "param": k},
                    {"token": _VAR.sub(lambda m: vars_.get(m.group(1), ""), v)},
                )
                continue
            pname = re.sub(r"\W", "_", k)[:40] or "param"
            if not re.match(r"^[a-zA-Z_]", pname):
                pname = "p_" + pname
            spec: dict[str, Any] = {
                "type": "int" if v.isdigit() else "string",
                "in": "query",
                "description": "",
            }
            if pname != k:
                spec["key"] = k
            if not _VAR.search(v) and v != "":
                spec["default"] = int(v) if v.isdigit() else v
            params[pname] = spec
        for h in req.get("header") or []:
            if not isinstance(h, dict) or h.get("disabled"):
                continue
            k, v = (
                str(h.get("key", "")),
                _VAR.sub(lambda m: vars_.get(m.group(1), ""), str(h.get("value", ""))),
            )
            if k.lower() == "authorization":
                scheme, _, secret = v.partition(" ")
                auth = (
                    {"type": "bearer"}
                    if scheme.lower() == "bearer"
                    else {"type": "header", "header": k, "prefix": scheme + " "}
                )
                creds = {"token": secret or v}
            elif _SECRET_NAME.search(k):
                auth, creds = {"type": "header", "header": k}, {"token": v}
            elif k.lower() in ("accept", "content-type") or k.lower().startswith("x-"):
                headers[k] = v
        oid = _slug(name.split("/")[-1])
        while oid in seen:
            oid = _slug(oid + "_2")
        seen.add(oid)
        desc = it.get("request", {}).get("description")
        op: dict[str, Any] = {
            "id": oid,
            "method": method,
            "summary": (desc if isinstance(desc, str) else "")[:200] or name,
            "ask_when": f"the question asks for {name.split('/')[-1].lower()}.",
            "path": "/" + "/".join(path_parts),
            "params": params,
            "response": {"rows": "", "line": "{value}", "empty": "nothing matched", "limit": 20},
        }
        if method == "POST":
            body = (req.get("body") or {}).get("raw") or "{}"
            body = _VAR.sub(lambda m: "{" + re.sub(r"\W", "_", m.group(1)) + "}", body)
            try:
                op["body"] = json.loads(body)
            except ValueError as exc:
                raise ManifestError(
                    f"{name}: the body is not JSON; only JSON search bodies are read"
                ) from exc
            for var in re.findall(r"\{(\w+)\}", json.dumps(op["body"])):
                params.setdefault(
                    var, {"type": "string", "in": "body", "required": True, "description": ""}
                )
            notes.append(
                f"{name} is a POST: mark it read-only only if it searches and changes nothing."
            )
        ops.append(op)
    if len(bases) > 1:
        notes.append(
            f"The requests use more than one host ({', '.join(sorted(bases))}); the first is the base URL."
        )
    if any(creds.values()):
        notes.append("Credentials were found; they go in the connection, not the connector file.")
    title = str((col.get("info") or {}).get("name") or "Connector")
    base = min(bases) if bases else ""
    return (
        {
            "format": FORMAT,
            "id": _slug(title, 30).replace("_", "-") or "connector",
            "name": title[:60],
            "kind": "live source",
            "colour": "#4f9186",
            "description": str((col.get("info") or {}).get("description") or "")[:600],
            "base_url": base,
            "clearance": "local",
            "auth": auth,
            "headers": headers,
            "operations": ops,
        },
        {k: v for k, v in creds.items() if v},
        notes,
    )


# --------------------------------------------------------------------- MCP


def _schema_param(schema: dict) -> dict:
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    spec: dict[str, Any] = {
        "description": str(schema.get("description") or schema.get("title") or "")[:200]
    }
    if schema.get("enum"):
        spec.update(type="enum", values=[str(v) for v in schema["enum"]][:50])
    elif t == "integer":
        spec["type"] = "int"
    elif t == "number":
        spec["type"] = "number"
    elif t == "boolean":
        spec["type"] = "bool"
    elif schema.get("format") in ("date", "date-time"):
        spec["type"] = "date"
    else:
        spec["type"] = "string"
    if "default" in schema and isinstance(schema["default"], (str, int, float, bool)):
        spec["default"] = schema["default"]
    return spec


_SCALAR = ("string", "integer", "number", "boolean")
# Fields that name a row, put first so each line starts with what it is.
_LEADS = ("name", "title", "id", "key", "label", "summary")


def _kind(schema: dict) -> str:
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "")
    return t or ("object" if schema.get("properties") else "")


def _fields_line(props: dict) -> str:
    """A line naming the row's plain fields: "{name} · status: {status} · …". Lists of
    plain values are joined; nested objects are left out (the author can add them)."""
    picked = []
    for k, ps in props.items():
        if not isinstance(ps, dict) or not re.match(r"^[A-Za-z_][\w\-]{0,63}$", k):
            continue
        kind = _kind(ps)
        if kind in _SCALAR:
            flt = "|date" if ps.get("format") in ("date", "date-time") else ""
            picked.append((k, flt))
        elif kind == "array" and _kind(ps.get("items") or {}) in _SCALAR:
            picked.append((k, "|join"))
    picked.sort(
        key=lambda kf: (
            kf[0].lower() not in _LEADS,
            _LEADS.index(kf[0].lower()) if kf[0].lower() in _LEADS else 0,
        )
    )
    picked = picked[:8]
    if not picked:
        return ""
    head, *rest = picked
    lead = (
        f"{{{head[0]}{head[1]}}}"
        if head[0].lower() in _LEADS
        else f"{head[0]}: {{{head[0]}{head[1]}}}"
    )
    return " · ".join([lead, *(f"{k}: {{{k}{f}}}" for k, f in rest)])


def _deref(node: Any, root: dict, depth: int = 0) -> Any:
    """A schema with its local references ("#/$defs/Ticket") written out, a few levels
    deep: generated schemas name every nested type that way."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            if depth >= 6:  # a type that contains itself: stop here
                return {}
            target: Any = root
            for part in ref[2:].split("/"):
                target = target.get(part, {}) if isinstance(target, dict) else {}
            return _deref(target, root, depth + 1)
        return {k: _deref(v, root, depth) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_deref(x, root, depth) for x in node]
    return node


def response_from_output_schema(schema: dict) -> dict | None:
    """A tool's declared output, as a response block that reads its structured result field
    by field: the rows are the first list of objects it returns (or the result itself), and
    each prints its plain fields. None when the schema says too little to draft from."""
    if not isinstance(schema, dict):
        return None
    schema = _deref(schema, schema)
    if _kind(schema) != "object":
        return None
    props = schema.get("properties") or {}
    if list(props) == ["result"] and _kind(props["result"]) in (*_SCALAR, ""):
        return None  # a plain value the SDK wrapped as {"result": ...}: read as text
    for k, ps in props.items():
        if not isinstance(ps, dict) or _kind(ps) != "array":
            continue
        item = ps.get("items") or {}
        if _kind(item) == "object" and (line := _fields_line(item.get("properties") or {})):
            return {
                "format": "json",
                "rows": k,
                "line": line,
                "empty": "nothing matched",
                "limit": 30,
                "summary": f"{{count}} {k.replace('_', ' ')}",
            }
    if line := _fields_line(props):
        return {"format": "json", "rows": "", "line": line, "empty": "nothing matched", "limit": 30}
    return None


_TEMPLATE_VAR = re.compile(r"\{(\w+)\}")


def draft_from_mcp(
    name: str,
    transport: dict,
    tools: list[dict],
    picks: list[str],
    *,
    resources: list[dict] | None = None,
    templates: list[dict] | None = None,
    resource_picks: list[str] | None = None,
) -> tuple[dict, list[str]]:
    """(draft manifest, notes) from what an MCP server offers. Only tools the server does
    not mark destructive or not-read-only can be picked; each becomes an operation whose
    parameters are the tool's declared arguments, and whose result is read field by field
    when the tool declares its output. A picked resource (by URI, or a template by its URI
    template) becomes an operation that reads it, its {placeholders} the parameters."""
    by = {t["name"]: t for t in tools}
    ops, notes, seen = [], [], set()

    def new_id(raw: str) -> str:
        oid = _slug(raw)
        while oid in seen:
            oid = _slug(oid + "_2")
        seen.add(oid)
        return oid

    for tname in picks:
        t = by.get(tname)
        if not t:
            raise ManifestError(f"the server has no tool {tname!r}")
        if t.get("refused"):
            raise ManifestError(f"{tname} cannot be a token's operation: {t['refused']}")
        if not t.get("read_only"):
            notes.append(
                f"{tname}: the server does not say it is read-only; include it only if it only looks things up."
            )
        schema = t.get("input_schema") or {}
        required = set(schema.get("required") or [])
        params = {}
        for pname, ps in (schema.get("properties") or {}).items():
            if not isinstance(ps, dict) or not re.match(r"^[a-zA-Z_][\w\-]{0,63}$", pname):
                continue
            spec = _schema_param(ps)
            if pname in required:
                spec["required"] = True
            params[pname] = spec
        response = response_from_output_schema(t.get("output_schema") or {})
        if response is None:
            response = {
                "format": "text",
                "rows": "",
                "line": "{text}",
                "empty": "nothing matched",
                "limit": 30,
            }
        ops.append(
            {
                "id": new_id(tname),
                "tool": tname,
                "read_only": True,
                "summary": t.get("description", "")[:300] or tname,
                "ask_when": f"the question calls for {tname.replace('_', ' ')}.",
                "params": params,
                "response": response,
            }
        )

    offered = {r["uri"]: r for r in resources or []}
    offered.update({t["uri_template"]: t for t in templates or []})
    for uri in resource_picks or []:
        r = offered.get(uri)
        if r is None:
            raise ManifestError(f"the server offers no resource {uri!r}")
        label = r.get("name") or uri
        params = {
            v: {"type": "string", "required": True, "description": f"the {v} in {uri}"}
            for v in dict.fromkeys(_TEMPLATE_VAR.findall(uri))
        }
        if "json" in (r.get("mime_type") or ""):
            notes.append(
                f"{label} is JSON: it is drafted as text; set its format to json, with rows "
                "and a line, to read it field by field."
            )
        ops.append(
            {
                "id": new_id(label),
                "resource": uri,
                "read_only": True,
                "summary": (r.get("description") or "")[:300] or f"reads {label}",
                "ask_when": f"the question needs what {label} says.",
                "params": params,
                "response": {
                    "format": "text",
                    "rows": "",
                    "line": "{text}",
                    "empty": "nothing there",
                    "limit": 40,
                },
            }
        )
    if not ops:
        raise ManifestError("pick at least one tool or resource")
    return (
        {
            "format": FORMAT,
            "id": _slug(name or "mcp-server", 30).replace("_", "-"),
            "name": (name or "MCP server")[:60],
            "kind": "MCP server",
            "colour": "#8a6bd1",
            "clearance": "local",
            "transport": transport,
            "auth": {"type": "none"}
            if transport.get("type") == "mcp_stdio"
            else {"type": "bearer"},
            "operations": ops,
        },
        notes,
    )
