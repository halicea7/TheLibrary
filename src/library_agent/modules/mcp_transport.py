"""Tokens whose source is an MCP server: the operation is a tool, called once per question.

Two ways to reach one. `mcp_stdio` starts the server on this machine -- a program and its
arguments, never a shell line -- with only PATH, HOME and the locale from this machine's
environment, the manifest's own variables, and the secret in the one variable the manifest
names. `mcp_http` talks to a server at a URL; the SDK follows a redirect only when it stays
on that origin.

Read-only is checked twice: the manifest's author declares every operation read-only, and
the server's own annotations are read before the call -- a tool it marks destructive, or
not read-only, is refused whatever the manifest says."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

from library_agent.modules.manifest import Module, Operation
from library_agent.modules.store import ModuleConfig

# The server's stderr is its own business, not the library's log.
_QUIET = open(os.devnull, "w")  # noqa: SIM115 -- one handle for the process's life
_KEEP_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "SYSTEMROOT")


class MCPError(Exception):
    pass


def server_env(module: Module, cfg: ModuleConfig) -> dict[str, str]:
    t = module.transport
    env = {k: os.environ[k] for k in _KEEP_ENV if k in os.environ}
    env.update(t.env)
    if t.secret_env:
        if not cfg.token:
            raise MCPError(f"{module.name}: no credentials configured")
        env[t.secret_env] = cfg.token
    return env


def _headers(module: Module, cfg: ModuleConfig) -> dict[str, str]:
    a = module.auth_spec()
    if a.type == "none":
        return {}
    if not cfg.token:
        raise MCPError(f"{module.name}: no credentials configured")
    if a.type == "bearer":
        return {a.header: f"Bearer {cfg.token}"}
    if a.type == "header":
        return {a.header: f"{a.prefix}{cfg.token}"}
    raise MCPError(f"{module.name}: an MCP server over HTTP signs in by bearer or header")


@asynccontextmanager
async def session(module: Module, cfg: ModuleConfig):
    from mcp import ClientSession

    t = module.transport
    timeout = module.limits.timeout
    if t.type == "mcp_stdio":
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=t.command, args=list(t.args), env=server_env(module, cfg)
        )
        async with (
            stdio_client(params, errlog=_QUIET) as (read, write),
            ClientSession(read, write, read_timeout_seconds=timeout) as s,
        ):
            await s.initialize()
            yield s
        return
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    url = (cfg.base_url or t.url).rstrip("/")
    if not url:
        raise MCPError(f"{module.name}: no server URL configured")
    client = create_mcp_http_client(headers=_headers(module, cfg))
    async with client, streamable_http_client(url, http_client=client) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write, read_timeout_seconds=timeout) as s:
            await s.initialize()
            yield s


def _leaf(exc: BaseException) -> str:
    """The first real error inside an exception group, named plainly."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


def _refused(tool) -> str | None:
    a = getattr(tool, "annotations", None)
    if a is None:
        return None
    if getattr(a, "destructive_hint", None) is True:
        return "the server marks it destructive"
    if getattr(a, "read_only_hint", None) is False:
        return "the server says it is not read-only"
    return None


async def list_tools(module: Module, cfg: ModuleConfig) -> list[dict]:
    """The server's tools, for drafting a token: name, description, input schema, and
    whether each may be used (read-only by the server's own account, or at least not
    marked destructive)."""
    try:
        async with asyncio.timeout(module.limits.timeout + 15):
            async with session(module, cfg) as s:
                tools = (await s.list_tools()).tools
    except MCPError:
        raise
    except Exception as exc:
        raise MCPError(
            f"{module.name}: could not reach the MCP server ({_leaf(exc)})"[:300]
        ) from exc
    out = []
    for t in tools:
        a = getattr(t, "annotations", None)
        out.append(
            {
                "name": t.name,
                "description": (t.description or "")[:400],
                "input_schema": t.input_schema or {},
                "read_only": bool(a and a.read_only_hint),
                "refused": _refused(t),
            }
        )
    return out


def _text(result) -> str:
    return "\n".join(
        getattr(c, "text", "") for c in (result.content or []) if getattr(c, "type", "") == "text"
    )


def rows_of(op: Operation, result, max_bytes: int) -> tuple[list, str]:
    """(rows, raw text) from a tool result: its structured content when it has any,
    otherwise its text -- as JSON when it parses, else one row per line."""
    from library_agent.modules.execute import dig

    doc: Any = getattr(result, "structured_content", None)
    if op.render.format == "text":
        # Read as text: the tool's own words, even when the server also wrapped them as
        # structured content ({"result": "..."}).
        doc = None
    raw = json.dumps(doc, ensure_ascii=False) if doc is not None else _text(result)
    if not raw and getattr(result, "structured_content", None) is not None:
        raw = json.dumps(result.structured_content, ensure_ascii=False)
    if len(raw.encode()) > max_bytes:
        raise MCPError(f"result over {max_bytes:,} bytes")
    if doc is None and op.render.format == "json":
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = None
    if doc is None:
        return [{"text": ln, "line": ln} for ln in raw.splitlines() if ln.strip()], raw
    rows = dig(doc, op.render.rows)
    if isinstance(rows, dict):
        rows = [rows]
    return (rows if isinstance(rows, list) else [] if rows is None else [rows]), raw


async def call_tool(
    module: Module, cfg: ModuleConfig, op: Operation, args: dict
) -> tuple[list, str]:
    arguments = {**op.const_query, **args}
    # Decided inside the session, raised after it closes: an exception inside the SDK's
    # task group comes out wrapped, and the reason would be lost.
    refusal, result = "", None
    try:
        async with asyncio.timeout(module.limits.timeout + 15), session(module, cfg) as s:
            tools = {t.name: t for t in (await s.list_tools()).tools}
            tool = tools.get(op.tool)
            if tool is None:
                refusal = f"{module.name}: the server has no tool {op.tool!r}"
            elif why := _refused(tool):
                refusal = f"{module.name}: refused {op.tool}: {why}"
            else:
                result = await s.call_tool(op.tool, arguments)
    except MCPError:
        raise
    except Exception as exc:
        raise MCPError(f"{module.name}: the MCP call failed ({_leaf(exc)})"[:300]) from exc
    if refusal:
        raise MCPError(refusal)
    if getattr(result, "is_error", False):
        raise MCPError(f"{module.name}: {op.tool} answered with an error: {_text(result)[:200]}")
    return rows_of(op, result, module.limits.max_bytes)
