"""Connectors: making, editing, trying, sharing and drafting the tokens' manifests.

A connector's manifest is data in `~/.library-agent/connectors/<id>.json`; its secret and
seating live in modules.json (see Settings › modules). Nothing here returns a secret.
Built-ins can be read and duplicated, not changed."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from library_agent.modules import registry
from library_agent.modules import store as mod_store
from library_agent.modules.manifest import ManifestError, Module, from_dict, to_dict

router = APIRouter(prefix="/api/connectors", tags=["connectors"])


def describe(m: Module, base_url: str = "") -> dict:
    """What a token can do, in the words of an approval: where it reaches, how it signs
    in, what it asks, and whether anything it does is not a plain read."""
    base = base_url or m.base_url or m.transport.url
    t = m.transport
    return {
        "id": m.id,
        "name": m.name,
        "transport": t.type,
        # Said plainly in an approval: this starts a program on your machine.
        "runs": " ".join([t.command, *t.args]) if t.type == "mcp_stdio" else None,
        "host": urlparse(base).hostname if base else None,
        "other_hosts": list(m.allowed_hosts),
        "auth": m.auth_spec().type,
        "clearance": "local models only" if m.local_only else "any model",
        "operations": [
            {"id": o.id, "method": o.method, "path": o.tool or o.path, "summary": o.summary}
            for o in m.operations
        ],
        "searches_by_post": sum(o.method == "POST" for o in m.operations),
        "limits": {
            "rate_per_minute": m.limits.rate_per_minute,
            "max_bytes": m.limits.max_bytes,
            "timeout": m.limits.timeout,
        },
    }


def _check(manifest: dict) -> Module:
    try:
        return from_dict(manifest)
    except ManifestError as exc:
        raise HTTPException(422, str(exc)) from exc


def _write(m: Module, manifest: dict) -> None:
    p = registry.path_for(m.id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(to_dict(m), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    registry.forget()


def _custom(mid: str) -> Module:
    m = registry.get(mid)
    if not m:
        raise HTTPException(404, f"no connector {mid!r}")
    if m.builtin:
        raise HTTPException(403, f"{m.name} is built in; duplicate it to change it")
    return m


class SaveIn(BaseModel):
    manifest: dict
    design: dict | None = None
    replace: bool = False


class TryIn(BaseModel):
    manifest: dict
    operation: str
    args: dict[str, Any] = {}
    # A connection to try with before it is saved; otherwise the saved one for this id.
    base_url: str | None = None
    token: str | None = None
    username: str | None = None


@router.get("")
async def list_connectors() -> dict:
    mods = registry.all_modules()
    return {
        "connectors": [
            {**to_dict(m), "builtin": m.builtin, "source": m.source} for m in mods.values()
        ],
        "errors": registry.errors(),
        "folder": str(registry.folder()),
    }


EXAMPLES = Path(__file__).resolve().parents[2] / "modules" / "examples"


def _examples() -> list[dict]:
    index = json.loads((EXAMPLES / "index.json").read_text())
    out = []
    for e in index["examples"]:
        m = json.loads((EXAMPLES / e["file"]).read_text())
        out.append({"slug": e["file"].removesuffix(".json"), "shows": e["shows"], "manifest": m})
    return out


@router.get("/examples")
async def examples() -> dict:
    """Working modules against public services, to start a new one from: each is loaded
    into the builder as a draft to try, change and make -- the guide is the example."""
    return {
        "examples": [
            {
                "slug": e["slug"],
                "name": e["manifest"]["name"],
                "kind": e["manifest"].get("kind", ""),
                "colour": e["manifest"].get("colour", ""),
                "description": e["manifest"].get("description", ""),
                "shows": e["shows"],
                "transport": (e["manifest"].get("transport") or {}).get("type", "http"),
                "operations": [o["id"] for o in e["manifest"]["operations"]],
                "taken": registry.get(e["manifest"]["id"]) is not None,
            }
            for e in _examples()
        ]
    }


@router.get("/examples/{slug}")
async def example(slug: str) -> dict:
    for e in _examples():
        if e["slug"] == slug:
            return {"manifest": e["manifest"], "shows": e["shows"]}
    raise HTTPException(404, "no such example")


@router.get("/{mid}")
async def get_connector(mid: str) -> dict:
    m = registry.get(mid)
    if not m:
        raise HTTPException(404, f"no connector {mid!r}")
    return {**to_dict(m), "builtin": m.builtin}


@router.post("/check")
async def check(body: SaveIn) -> dict:
    """Whether a manifest would load, and what it could do -- without saving it."""
    m = _check(body.manifest)
    return {"ok": True, "summary": describe(m)}


@router.post("")
async def save(body: SaveIn) -> dict:
    """Create a connector (or, with replace, overwrite your own of the same id)."""
    m = _check(body.manifest)
    existing = registry.get(m.id)
    if existing and existing.builtin:
        raise HTTPException(409, f"{m.id!r} is a built-in's id; choose another")
    if existing and not body.replace:
        raise HTTPException(409, f"a connector {m.id!r} already exists")
    _write(m, body.manifest)
    if body.design is not None:
        from library_agent.modules.design import DesignError, clean

        try:
            design = clean(body.design)
        except DesignError as exc:
            raise HTTPException(422, str(exc)) from exc
        cfgs = mod_store.load()
        cfg = cfgs.get(m.id) or mod_store.ModuleConfig(id=m.id)
        cfg.design = design
        cfgs[m.id] = cfg
        mod_store.save(cfgs)
    return {"ok": True, "summary": describe(m)}


@router.put("/{mid}")
async def edit(mid: str, body: SaveIn) -> dict:
    """Replace your connector. Renaming its id moves its connection and seat with it."""
    _custom(mid)
    m = _check(body.manifest)
    if m.id != mid:
        other = registry.get(m.id)
        if other:
            raise HTTPException(409, f"a connector {m.id!r} already exists")
    _write(m, body.manifest)
    if m.id != mid:
        registry.path_for(mid).unlink(missing_ok=True)
        cfgs = mod_store.load()
        if mid in cfgs:
            cfg = cfgs.pop(mid)
            cfg.id = m.id
            cfgs[m.id] = cfg
            mod_store.save(cfgs)
        registry.forget()
    return {"ok": True, "summary": describe(m)}


@router.delete("/{mid}")
async def remove(mid: str) -> dict:
    """Delete your connector, and its connection (the secret goes with it)."""
    m = _custom(mid)
    registry.path_for(mid).unlink(missing_ok=True)
    registry.forget()
    cfgs = mod_store.load()
    if cfgs.pop(mid, None) is not None:
        mod_store.save(cfgs)
    return {"ok": True, "removed": m.name}


@router.post("/{mid}/duplicate")
async def duplicate(mid: str) -> dict:
    """A copy to change -- of a built-in or your own. The copy has no connection."""
    m = registry.get(mid)
    if not m:
        raise HTTPException(404, f"no connector {mid!r}")
    d = to_dict(m)
    base = re.sub(r"-copy\d*$", "", m.id)[:32]
    n, new = 1, f"{base}-copy"
    while registry.get(new):
        n += 1
        new = f"{base}-copy{n}"
    d.update(id=new, name=f"{m.name} (copy)"[:60])
    copy = _check(d)
    _write(copy, d)
    return {"ok": True, "id": new, "summary": describe(copy)}


@router.post("/try")
async def try_it(body: TryIn) -> dict:
    """Run one operation of an unsaved draft once: the raw response beside the rendered
    rows, so a path or a line format can be fixed before anything is saved. The
    connection given here is used for this call only and not kept."""
    from library_agent.modules.execute import ModuleError, call
    from library_agent.modules.render import render_rows

    m = _check(body.manifest)
    op = m.op(body.operation)
    if not op:
        raise HTTPException(404, f"no operation {body.operation!r}")
    saved = mod_store.config_for(m.id)
    base = (body.base_url or saved.base_url or m.base_url).rstrip("/")
    # The saved secret goes only to the host it was saved for: a draft that borrows an
    # existing id cannot send that connector's token somewhere else.
    same_host = (
        bool(saved.base_url) and urlparse(saved.base_url).hostname == urlparse(base).hostname
    )
    if m.is_mcp:
        # No host to compare: the saved secret goes only to the very server it was saved
        # for -- the same command, arguments and environment, or the same URL.
        known = registry.get(m.id)
        same_host = bool(known) and known.transport == m.transport
    cfg = mod_store.ModuleConfig(
        id=m.id,
        base_url=base,
        token=body.token or (saved.token if same_host else ""),
        username=body.username
        if body.username is not None
        else (saved.username if same_host else ""),
    )
    try:
        res = await call(m, cfg, op, body.args, debug=True)
    except ModuleError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "count": res["count"],
        "pages": res["pages"],
        "rendered": render_rows(op, res["rows"]),
        "rows": res["rows"][:5],
        "sample": res.get("sample", ""),
    }


@router.get("/{mid}/export")
async def export(mid: str) -> dict:
    """A module as a file to share: its overall configuration and its look -- never its
    connection, and nothing of this installation (see manifest.for_sharing). `requires`
    names what whoever imports it must fill in."""
    from library_agent.modules.manifest import for_sharing

    m = registry.get(mid)
    if not m:
        raise HTTPException(404, f"no connector {mid!r}")
    manifest, requires = for_sharing(m)
    design = mod_store.config_for(mid).design or {}
    return {"library_module": 1, "manifest": manifest, "design": design, "requires": requires}


class ImportIn(BaseModel):
    file: dict
    approve: bool = False


@router.post("/import")
async def import_token(body: ImportIn) -> dict:
    """A shared token file. Without approve, it is only described -- where it reaches, how
    it signs in, what it can ask -- so nothing arrives unseen; with approve, it is saved."""
    f = body.file
    manifest = f.get("manifest") if isinstance(f.get("manifest"), dict) else f
    m = _check(manifest)
    summary = describe(m)
    if isinstance(f.get("requires"), dict) and f["requires"]:
        summary["requires"] = f["requires"]
    if registry.get(m.id):
        summary["conflict"] = f"a connector {m.id!r} already exists; change the id to import it"
    if not body.approve:
        return {"ok": True, "approved": False, "summary": summary}
    if summary.get("conflict"):
        raise HTTPException(409, summary["conflict"])
    return await save(SaveIn(manifest=manifest, design=f.get("design")))


class OpenAPIIn(BaseModel):
    spec: str
    pick: list[str] | None = None
    name: str = ""
    base_url: str = ""


@router.post("/draft/openapi")
async def draft_openapi(body: OpenAPIIn) -> dict:
    """From an OpenAPI/Swagger document: without `pick`, the GET operations it declares;
    with `pick`, a draft manifest of those."""
    from library_agent.modules import importers

    if len(body.spec) > 5_000_000:
        raise HTTPException(413, "that document is over 5 MB")
    try:
        spec = importers.load_spec(body.spec)
        if not body.pick:
            ops = importers.openapi_operations(spec)
            return {
                "title": (spec.get("info") or {}).get("title", ""),
                "base_url": importers._base_url(spec),
                "auth": importers._auth(spec),
                "operations": ops[:500],
                "total": len(ops),
            }
        draft = importers.draft_from_openapi(
            spec, body.pick, name=body.name, base_url=body.base_url
        )
    except ManifestError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"manifest": draft}


class CurlIn(BaseModel):
    command: str


@router.post("/draft/curl")
async def draft_curl(body: CurlIn) -> dict:
    """From a working curl command: a one-operation draft, and any credentials found in
    it (for the connection form, never the file)."""
    from library_agent.modules import importers

    try:
        manifest, creds, notes = importers.draft_from_curl(body.command[:20_000])
    except ManifestError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"manifest": manifest, "credentials": creds, "notes": notes}


class PostmanIn(BaseModel):
    collection: str
    pick: list[str] | None = None


@router.post("/draft/postman")
async def draft_postman(body: PostmanIn) -> dict:
    """From a Postman collection (v2.x): without `pick`, its requests; with `pick`, a
    draft manifest of those, credentials found (for the connection), and notes."""
    from library_agent.modules import importers

    if len(body.collection) > 5_000_000:
        raise HTTPException(413, "that collection is over 5 MB")
    try:
        col = importers.load_postman(body.collection)
        if not body.pick:
            return {
                "title": (col.get("info") or {}).get("name", ""),
                "requests": importers.postman_requests(col)[:500],
            }
        manifest, creds, notes = importers.draft_from_postman(col, body.pick)
    except ManifestError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"manifest": manifest, "credentials": creds, "notes": notes}


class MCPIn(BaseModel):
    name: str = ""
    transport: dict
    token: str | None = None  # for this probe only; saved with the connection, not here
    pick: list[str] | None = None


@router.post("/draft/mcp")
async def draft_mcp(body: MCPIn) -> dict:
    """From an MCP server: without `pick`, its tools (and which may be used: the server's
    own read-only / destructive marks); with `pick`, a draft manifest of those. Starting a
    stdio server runs its command on this machine -- the command is shown before this is
    called, and nothing is kept."""
    from library_agent.modules import importers
    from library_agent.modules.mcp_transport import MCPError, list_tools

    probe_manifest = {
        "id": "probe",
        "name": body.name or "MCP server",
        "transport": body.transport,
        "auth": {"type": "bearer"}
        if body.token and body.transport.get("type") == "mcp_http"
        else {"type": "none"},
        "operations": [{"id": "probe", "tool": "probe", "read_only": True}],
    }
    probe = _check(probe_manifest)
    cfg = mod_store.ModuleConfig(id="probe", token=body.token or "")
    try:
        tools = await list_tools(probe, cfg)
        if not body.pick:
            return {"tools": tools}
        manifest, notes = importers.draft_from_mcp(body.name, body.transport, tools, body.pick)
    except (MCPError, ManifestError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"manifest": manifest, "notes": notes}
