"""Tokens whose source is an MCP server: an operation is a tool it calls, or a resource it
reads (a document the server offers, by URI or URI template).

Two ways to reach one. `mcp_stdio` starts the server on this machine -- a program and its
arguments, never a shell line -- with only PATH, HOME and the locale from this machine's
environment, the manifest's own variables, and the secret in the one variable the manifest
names. `mcp_http` talks to a server at a URL; the SDK follows a redirect only when it stays
on that origin.

Read-only is checked twice: the manifest's author declares every operation read-only, and
the server's own annotations are read before every call -- a tool it marks destructive, or
not read-only, is refused whatever the manifest says. Resources are read-only by the
protocol: reading one changes nothing.

A server is kept running between calls (one session per module and configuration, closed
after `mcp_idle_seconds` unused, restarted after any failure), so a stdio server is started
once, not once per question. Calls to one server queue and run one at a time. The tool list
is fetched again over the open session before each call, so the read-only check never
rests on a stale listing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any
from urllib.parse import quote

from library_agent.config import settings
from library_agent.modules.manifest import Module, Operation
from library_agent.modules.store import ModuleConfig

# The server's stderr is its own business, not the library's log.
_QUIET = open(os.devnull, "w")  # noqa: SIM115 -- one handle for the process's life
_KEEP_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "SYSTEMROOT")


class MCPError(Exception):
    pass


class _Ended(MCPError):
    """A call queued on a session that closed before reaching it: safe to send again."""


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
            # What the server said it offers, kept for the listing of resources.
            s.library_caps = (await s.initialize()).capabilities
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
            # What the server said it offers, kept for the listing of resources.
            s.library_caps = (await s.initialize()).capabilities
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


async def _all(fetch, field: str) -> list:
    """Every page of a listing."""
    out, cursor = [], None
    for _ in range(50):  # a guard against a server that never stops paging
        page = await fetch(params={"cursor": cursor} if cursor else None)
        out.extend(getattr(page, field) or [])
        cursor = getattr(page, "next_cursor", None)
        if not cursor:
            break
    return out


async def _resources(s) -> tuple[list, list]:
    """(resources, templates), or empty when the server offers none."""
    caps = getattr(s, "library_caps", None)
    if caps is not None and getattr(caps, "resources", None) is None:
        return [], []
    try:
        res = await _all(s.list_resources, "resources")
    except Exception:  # noqa: BLE001 -- a server without resources may just say so
        res = []
    try:
        tpl = await _all(s.list_resource_templates, "resource_templates")
    except Exception:  # noqa: BLE001
        tpl = []
    return res, tpl


async def list_offer(module: Module, cfg: ModuleConfig) -> dict[str, list[dict]]:
    """What the server offers, for drafting a token: its tools (with whether each may be
    used -- read-only by the server's own account, or at least not marked destructive),
    its resources, and its resource templates. A fresh session: this is a probe of a
    configuration not yet saved, so nothing is kept."""
    try:
        async with asyncio.timeout(module.limits.timeout + 15):
            async with session(module, cfg) as s:
                tools = await _all(s.list_tools, "tools")
                resources, templates = await _resources(s)
    except MCPError:
        raise
    except Exception as exc:
        raise MCPError(
            f"{module.name}: could not reach the MCP server ({_leaf(exc)})"[:300]
        ) from exc
    out_tools = []
    for t in tools:
        a = getattr(t, "annotations", None)
        out_tools.append(
            {
                "name": t.name,
                "description": (t.description or "")[:400],
                "input_schema": t.input_schema or {},
                "output_schema": getattr(t, "output_schema", None) or {},
                "read_only": bool(a and a.read_only_hint),
                "refused": _refused(t),
            }
        )
    return {
        "tools": out_tools,
        "resources": [
            {
                "uri": str(r.uri),
                "name": r.name,
                "description": (r.description or "")[:400],
                "mime_type": r.mime_type or "",
            }
            for r in resources
        ],
        "templates": [
            {
                "uri_template": str(t.uri_template),
                "name": t.name,
                "description": (t.description or "")[:400],
                "mime_type": t.mime_type or "",
            }
            for t in templates
        ],
    }


async def list_tools(module: Module, cfg: ModuleConfig) -> list[dict]:
    return (await list_offer(module, cfg))["tools"]


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


# ------------------------------------------------------------------ resources

_VAR = re.compile(r"\{(\w+)\}")


def resource_uri(op: Operation, args: dict) -> str:
    """The resource's URI with the values filled in, each escaped so it stays one segment."""
    values = {**op.const_query, **args}

    def one(m: re.Match) -> str:
        v = values.get(m.group(1))
        if v is None or v == "":
            raise MCPError(f"{op.id}: no value for {{{m.group(1)}}}")
        return quote(str(v), safe="")

    return _VAR.sub(one, op.resource)


def rows_of_resource(op: Operation, result, max_bytes: int) -> tuple[list, str]:
    """(rows, raw text) from a resource read. As text, a row per paragraph, so a document
    prints whole up to the line limit; as JSON, the rows at the operation's path. Binary
    contents are not read."""
    from library_agent.modules.execute import dig

    parts = [c for c in (result.contents or []) if getattr(c, "text", None) is not None]
    raw = "\n\n".join(c.text for c in parts)
    if len(raw.encode()) > max_bytes:
        raise MCPError(f"result over {max_bytes:,} bytes")
    if op.render.format == "json":
        docs = []
        for c in parts:
            try:
                docs.append(json.loads(c.text))
            except ValueError:
                pass
        if docs:
            doc = docs[0] if len(docs) == 1 else docs
            rows = dig(doc, op.render.rows)
            if isinstance(rows, dict):
                rows = [rows]
            return (rows if isinstance(rows, list) else [] if rows is None else [rows]), raw
    rows = [
        {"text": para.strip(), "uri": str(c.uri)}
        for c in parts
        for para in re.split(r"\n\s*\n", c.text)
        if para.strip()
    ]
    return rows, raw


# ------------------------------------------------------------------ live sessions


def _key(module: Module, cfg: ModuleConfig) -> str:
    """Everything that decides which server and how it is reached. A change to any of it
    is a different session; the secret is hashed, not held in the key."""
    t, a = module.transport, module.auth_spec()
    blob = json.dumps(
        [
            module.id,
            t.type,
            t.command,
            list(t.args),
            sorted(t.env.items()),
            t.secret_env,
            t.url,
            cfg.base_url,
            a.type,
            a.header,
            a.prefix,
            module.limits.timeout,
            hashlib.sha256((cfg.token or "").encode()).hexdigest(),
        ],
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


Job = Callable[[Any], Awaitable[Any]]


class _Live:
    """One open session, owned by one task: the SDK's session must be entered and left in
    the same task, so callers hand it work through a queue rather than sharing it."""

    def __init__(self, module: Module, cfg: ModuleConfig, key: str):
        self.key, self.module_id = key, module.id
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue[tuple[Job, asyncio.Future]] = asyncio.Queue()
        self.closed = False
        self.task = self.loop.create_task(self._own(module, cfg))

    async def _own(self, module: Module, cfg: ModuleConfig) -> None:
        failure: BaseException | None = None
        opened = False
        try:
            async with session(module, cfg) as s:
                opened = True
                while True:
                    try:
                        job, fut = await asyncio.wait_for(
                            self.queue.get(), settings().mcp_idle_seconds
                        )
                    except TimeoutError:
                        self.closed = True  # unused for a while: let the server go
                        return
                    if fut.done():  # its caller gave up waiting
                        continue
                    try:
                        fut.set_result(await job(s))
                    except Exception as exc:  # noqa: BLE001
                        # A failed call may have left the session broken: start afresh.
                        # Closed before the caller hears, so its next call opens a new one.
                        self.closed = True
                        if not fut.done():
                            fut.set_exception(exc)
                        return
        except asyncio.CancelledError:
            pass
        except BaseException as exc:  # noqa: BLE001 -- the server would not start, or went away
            failure = exc
        finally:
            self.closed = True
            if _pool.get(self.key) is self:
                del _pool[self.key]
            while not self.queue.empty():
                _, fut = self.queue.get_nowait()
                if fut.done():
                    continue
                if not opened and failure is not None:
                    # The server never started: that is the answer, not worth a retry.
                    fut.set_exception(
                        MCPError(
                            f"{module.name}: could not start the MCP server ({_leaf(failure)})"[
                                :300
                            ]
                        )
                    )
                else:
                    fut.set_exception(_Ended(f"{module.name}: the MCP session ended"))

    async def run(self, job: Job, timeout: float) -> Any:
        fut: asyncio.Future = self.loop.create_future()
        self.queue.put_nowait((job, fut))
        try:
            async with asyncio.timeout(timeout):
                return await fut
        except TimeoutError:
            # A server that does not answer is not trusted with the next call either.
            self.task.cancel()
            raise


_pool: dict[str, _Live] = {}


def _live(module: Module, cfg: ModuleConfig) -> _Live:
    key = _key(module, cfg)
    loop = asyncio.get_running_loop()
    cur = _pool.get(key)
    if cur is not None and not cur.closed and cur.loop is loop:
        return cur
    # The same module under an older configuration is closed now, not left to idle out.
    for other in [x for x in _pool.values() if x.module_id == module.id and x.key != key]:
        if other.loop is loop:
            other.task.cancel()
        _pool.pop(other.key, None)
    cur = _pool[key] = _Live(module, cfg, key)
    return cur


async def close_all() -> None:
    """Close every server this process keeps open (on shutdown)."""
    loop = asyncio.get_running_loop()
    mine = [x for x in _pool.values() if x.loop is loop]
    for x in mine:
        x.task.cancel()
    for x in mine:
        with suppress(BaseException):
            await x.task
    _pool.clear()


def live_sessions() -> int:
    return sum(1 for x in _pool.values() if not x.closed)


async def call_operation(
    module: Module, cfg: ModuleConfig, op: Operation, args: dict
) -> tuple[list, str]:
    """One operation over the module's live session: a tool call, checked against the
    server's own marks first, or a resource read."""
    if op.resource:
        uri = resource_uri(op, args)

        async def read(s):
            return await s.read_resource(uri)

        result = await _run(module, cfg, read, "the MCP read failed")
        return rows_of_resource(op, result, module.limits.max_bytes)

    arguments = {**op.const_query, **args}

    async def use(s):
        # Decided inside the session, raised by the caller: an exception inside the SDK's
        # task group comes out wrapped, and the reason would be lost.
        tools = {t.name: t for t in await _all(s.list_tools, "tools")}
        tool = tools.get(op.tool)
        if tool is None:
            return f"{module.name}: the server has no tool {op.tool!r}", None
        if why := _refused(tool):
            return f"{module.name}: refused {op.tool}: {why}", None
        return "", await s.call_tool(op.tool, arguments)

    refusal, result = await _run(module, cfg, use, "the MCP call failed")
    if refusal:
        raise MCPError(refusal)
    if getattr(result, "is_error", False):
        raise MCPError(f"{module.name}: {op.tool} answered with an error: {_text(result)[:200]}")
    return rows_of(op, result, module.limits.max_bytes)


async def _run(module: Module, cfg: ModuleConfig, job: Job, what: str) -> Any:
    try:
        try:
            return await _live(module, cfg).run(job, module.limits.timeout + 15)
        except _Ended:
            # Queued as the session closed (idle, or after a failure): once more, fresh.
            return await _live(module, cfg).run(job, module.limits.timeout + 15)
    except MCPError:
        raise
    except Exception as exc:
        raise MCPError(f"{module.name}: {what} ({_leaf(exc)})"[:300]) from exc
