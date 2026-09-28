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
