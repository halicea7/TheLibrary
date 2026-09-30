"""The connectors API: make, edit, rename, duplicate, delete, share and draft tokens,
without ever returning or misdirecting a secret."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import HTTPException

from library_agent.api.routes import connectors as api
from library_agent.modules import execute, registry
from library_agent.modules import store as mod_store


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRARY_CONNECTORS_DIR", str(tmp_path / "connectors"))
    monkeypatch.setenv("LIBRARY_MODULES_FILE", str(tmp_path / "modules.json"))
    registry.forget()
    mod_store._cache = None
    yield tmp_path
    registry.forget()
    mod_store._cache = None


def manifest(mid="acme", base="https://api.acme.test"):
    return {
        "id": mid,
        "name": "Acme",
        "base_url": base,
        "auth": {"type": "bearer"},
        "operations": [
            {
                "id": "tickets",
                "summary": "open tickets",
                "ask_when": "tickets",
                "path": "/tickets",
                "response": {"rows": "data", "line": "{title}"},
            }
        ],
    }


async def test_make_edit_rename_duplicate_and_delete(home):
    await api.save(api.SaveIn(manifest=manifest(), design={"color": "#123456", "name": "ACME"}))
    assert registry.get("acme").name == "Acme"
    assert mod_store.config_for("acme").design == {"color": "#123456", "name": "ACME"}
    with pytest.raises(HTTPException) as e:
        await api.save(api.SaveIn(manifest=manifest()))
    assert e.value.status_code == 409  # exists; replace is explicit
    # a connection, then a rename: the connection moves with it
    cfgs = mod_store.load()
    cfgs["acme"].token = "s3cret"  # the design created its entry
    mod_store.save(cfgs)
    await api.edit("acme", api.SaveIn(manifest=manifest(mid="acme-2")))
    assert registry.get("acme") is None and registry.get("acme-2")
    assert mod_store.config_for("acme-2").token == "s3cret"
    dup = await api.duplicate("acme-2")
    assert dup["id"] == "acme-2-copy" and mod_store.config_for("acme-2-copy").token == ""
    await api.remove("acme-2")
    assert registry.get("acme-2") is None and "acme-2" not in mod_store.load()
    listed = await api.list_connectors()
    assert "s3cret" not in json.dumps(listed)


async def test_the_built_in_can_be_copied_not_changed(home):
    with pytest.raises(HTTPException) as e:
        await api.remove("sentinelone")
    assert e.value.status_code == 403
    with pytest.raises(HTTPException) as e:
        await api.save(api.SaveIn(manifest=manifest(mid="sentinelone")))
    assert e.value.status_code == 409
    dup = await api.duplicate("sentinelone")
    assert dup["id"] == "sentinelone-copy" and len(dup["summary"]["operations"]) == len(
        registry.get("sentinelone").operations
    )


async def test_an_import_is_described_before_it_is_approved(home):
    file = {"library_token": 1, "manifest": manifest(), "design": {"color": "#abcdef"}}
    seen = await api.import_token(api.ImportIn(file=file))
    assert seen["approved"] is False and registry.get("acme") is None
    s = seen["summary"]
    assert (
        s["host"] == "api.acme.test"
        and s["auth"] == "bearer"
        and s["operations"][0]["method"] == "GET"
    )
    await api.import_token(api.ImportIn(file=file, approve=True))
    assert registry.get("acme") and mod_store.config_for("acme").design == {"color": "#abcdef"}
    exported = await api.export("acme")
    assert exported["manifest"]["id"] == "acme" and "token" not in json.dumps(
        exported["manifest"]
    ).replace("token_url", "")


async def test_try_never_sends_a_saved_secret_to_another_host(home, monkeypatch):
    await api.save(api.SaveIn(manifest=manifest()))
    cfgs = mod_store.load()
    cfgs["acme"] = mod_store.ModuleConfig(
        id="acme", base_url="https://api.acme.test", token="s3cret"
    )
    mod_store.save(cfgs)
    seen = {}

    def transport(request: httpx.Request):
        seen["host"], seen["auth"] = request.url.host, request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"title": "one"}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        execute.httpx,
        "AsyncClient",
        lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(transport)}),
    )
    ok = await api.try_it(api.TryIn(manifest=manifest(), operation="tickets"))
    assert ok["ok"] and ok["rendered"] == "one" and seen["auth"] == "Bearer s3cret"
    evil = await api.try_it(
        api.TryIn(
            manifest=manifest(base="https://evil.test"),
            operation="tickets",
            base_url="https://evil.test",
        )
    )
    assert not evil["ok"] and "no credentials" in evil["error"]  # the saved token stayed home


async def test_drafts_from_openapi_and_curl():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Pets"},
        "servers": [{"url": "https://pets.test/v1"}],
        "components": {
            "securitySchemes": {"k": {"type": "apiKey", "in": "header", "name": "X-Key"}},
            "schemas": {
                "Pet": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "status": {"type": "string"},
                        "tags": {"type": "array"},
                    },
                }
            },
        },
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "listPets",
                    "summary": "List pets",
                    "parameters": [
                        {
                            "name": "status",
                            "in": "query",
                            "schema": {"type": "string", "enum": ["a", "b"]},
                        }
                    ],
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "items": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/Pet"},
                                            }
                                        },
                                    }
                                }
                            }
                        }
                    },
                },
                "post": {"summary": "make a pet"},
            },
            "/pets/{pet-id}": {
                "get": {
                    "parameters": [
                        {
                            "name": "pet-id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}
                            }
                        }
                    },
                }
            },
        },
    }
    listed = await api.draft_openapi(api.OpenAPIIn(spec=json.dumps(spec)))
    assert [o["key"] for o in listed["operations"]] == [
        "GET /pets",
        "GET /pets/{pet-id}",
    ]  # no POST
    assert listed["auth"] == {"type": "header", "header": "X-Key"}
    draft = (
        await api.draft_openapi(
            api.OpenAPIIn(spec=json.dumps(spec), pick=["GET /pets", "GET /pets/{pet-id}"])
        )
    )["manifest"]
    m = api._check(draft)
    assert m.base_url == "https://pets.test/v1"
    pets = m.op("listpets")
    assert pets.render.rows == "items" and pets.render.line == "{name} — {status}"
    one = m.operations[1]
    assert one.path == "/pets/{pet_id}" and one.params["required"] == ["pet_id"]

    got = await api.draft_curl(
        api.CurlIn(
            command="curl -H 'Authorization: ApiToken abc' 'https://s1.test/web/api/v2.1/agents?limit=10&computerName__contains=web'"
        )
    )
    d, creds = got["manifest"], got["credentials"]
    assert d["auth"] == {
        "type": "header",
        "header": "Authorization",
        "prefix": "ApiToken ",
    } and creds == {"token": "abc"}
    assert "abc" not in json.dumps(d)
    op = api._check(d).operations[0]
    assert op.path == "/web/api/v2.1/agents" and set(op.params["properties"]) == {
        "limit",
        "computerName__contains",
    }
    post = await api.draft_curl(
        api.CurlIn(
            command="""curl -X POST https://es.test/idx/_search -d '{"query":{"match_all":{}}}'"""
        )
    )
    assert post["notes"] and "read_only_post" in post["notes"][0]
    with pytest.raises(HTTPException):
        api._check(post["manifest"])  # a POST will not save until marked read-only
    with pytest.raises(HTTPException) as e:
        await api.draft_curl(api.CurlIn(command="curl -X DELETE https://x.test/a"))
    assert "changes data" in e.value.detail


async def test_the_examples_are_valid_modules_and_each_teaches_something():
    from library_agent.api.routes import connectors
    from library_agent.modules.manifest import from_dict

    listed = (await connectors.examples())["examples"]
    assert {x["slug"] for x in listed} >= {"openalex", "nvd", "arxiv", "github", "mcp-time"}
    for x in listed:
        assert x["shows"] and x["operations"] and x["description"]
        m = from_dict((await connectors.example(x["slug"]))["manifest"])
        assert m.operations and not m.local_only  # public data: any model may read it
    # An example ships no credentials: it signs in with nothing, or an MCP server's
    # secret variable is left for the person to fill.
    for x in listed:
        man = (await connectors.example(x["slug"]))["manifest"]
        assert (man.get("auth") or {"type": "none"}) == {"type": "none"}
        assert not (man.get("transport") or {}).get("env")
