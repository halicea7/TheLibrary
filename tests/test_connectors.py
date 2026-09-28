"""Connectors as data: manifests checked field by field, every auth scheme, every kind of
pagination, JSON/XML/CSV/text responses, and the guard rails -- all against an in-process
mock API, so nothing here touches the network."""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from library_agent.modules import execute, registry
from library_agent.modules.execute import ModuleError, call
from library_agent.modules.manifest import ManifestError, from_dict, to_dict
from library_agent.modules.render import format_row, render_rows
from library_agent.modules.store import ModuleConfig, is_configured

ITEMS = [
    {"id": i, "title": f"item {i}", "user": {"login": f"u{i}"}, "at": "2026-09-27T14:03:00Z"}
    for i in range(1, 8)
]


def mock_api() -> FastAPI:
    app = FastAPI()
    app.state.seen = []

    def auth_ok(request: Request, expect: str) -> bool:
        app.state.seen.append((request.url.path, dict(request.query_params), dict(request.headers)))
        return expect == "ok"

    @app.get("/bearer/items")
    async def bearer(request: Request, authorization: str | None = Header(default=None)):
        auth_ok(request, "ok")
        if authorization != "Bearer k1":
            return JSONResponse({"error": "no"}, status_code=401)
        return {"data": {"items": ITEMS[:2]}}

    @app.get("/query/items")
    async def query_key(request: Request, api_key: str | None = Query(default=None)):
        auth_ok(request, "ok")
        return {"data": ITEMS[:1]} if api_key == "k2" else JSONResponse({}, status_code=403)

    @app.get("/basic/items")
    async def basic(request: Request, authorization: str | None = Header(default=None)):
        auth_ok(request, "ok")
        want = "Basic " + base64.b64encode(b"alice:pw").decode()
        return ITEMS[:3] if authorization == want else JSONResponse({}, status_code=401)

    @app.post("/oauth/token")
    async def token(request: Request, authorization: str | None = Header(default=None)):
        app.state.tokens = getattr(app.state, "tokens", 0) + 1
        want = "Basic " + base64.b64encode(b"client:secret").decode()
        if authorization != want:
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        return {"access_token": "t-oauth", "expires_in": 3600}

    @app.get("/oauth/items")
    async def oauth_items(authorization: str | None = Header(default=None)):
        return ITEMS[:1] if authorization == "Bearer t-oauth" else JSONResponse({}, status_code=401)

    @app.get("/repos/{owner}/{repo}/issues")
    async def issues(owner: str, repo: str, state: str = "open"):
        return [{"title": f"{owner}/{repo} #{i}", "state": state} for i in range(2)]

    @app.get("/paged")
    async def paged(page: int = 1, per_page: int = 3):
        start = (page - 1) * per_page
        return {"items": ITEMS[start : start + per_page]}

    @app.get("/offset")
    async def offset(offset: int = 0, limit: int = 3):
        return {"items": ITEMS[offset : offset + limit]}

    @app.get("/cursor")
    async def cursor(after: int = 0):
        chunk = ITEMS[after : after + 3]
        nxt = after + 3 if after + 3 < len(ITEMS) else None
        return {"items": chunk, "meta": {"next": nxt}}

    @app.get("/linked")
    async def linked(request: Request, p: int = 1):
        chunk = ITEMS[(p - 1) * 3 : p * 3]
        headers = {}
        if p * 3 < len(ITEMS):
            headers["Link"] = f'<http://api.test/linked?p={p + 1}>; rel="next"'
        return JSONResponse({"items": chunk}, headers=headers)

    @app.get("/linked-off")
    async def linked_off():
        return JSONResponse(
            {"items": ITEMS[:1]}, headers={"Link": '<http://evil.test/x?p=2>; rel="next"'}
        )

    @app.get("/feed.xml")
    async def feed():
        xml = "<rss><channel><item><title>A</title><link>http://a</link></item><item><title>B</title><link>http://b</link></item></channel></rss>"
        return Response(xml, media_type="application/xml")

    @app.get("/bomb.xml")
    async def bomb():
        xml = '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;">]><r>&lol2;</r>'
        return Response(xml, media_type="application/xml")

    @app.get("/table.csv")
    async def table():
        return PlainTextResponse("host,os\nweb-1,linux\ndb-1,bsd\n")

    @app.get("/log.txt")
    async def log():
        return PlainTextResponse("first line\n\nsecond line\n")

    @app.post("/search")
    async def search(request: Request):
        body = await request.json()
        app.state.body = body
        return {
            "hits": {"hits": [{"_source": {"title": f"hit for {body['query']['match']['title']}"}}]}
        }

    @app.get("/big")
    async def big():
        return {"data": ["x" * 1000] * 50}

    return app


@pytest.fixture
def api(monkeypatch):
    app = mock_api()
    real = httpx.AsyncClient

    def client(*a, **kw):
        if "api.test" in str(kw.get("base_url", "")):
            kw["transport"] = httpx.ASGITransport(app=app)
        return real(*a, **kw)

    monkeypatch.setattr(execute.httpx, "AsyncClient", client)
    execute._cache.clear()
    execute._calls.clear()
    execute._oauth.clear()
    return app


def manifest(**over) -> dict:
    op = {
        "id": "items",
        "summary": "items",
        "ask_when": "asked about items",
        "path": "/bearer/items",
        "response": {"rows": "data.items[*]", "line": "{title} by {user.login} ({at|date})"},
    }
    op.update(over.pop("op", {}))
    d = {"id": "mock", "name": "Mock", "auth": {"type": "bearer"}, "operations": [op]}
    d.update(over)
    return d


def cfg(token="k1", username=""):
    return ModuleConfig(
        id="mock", base_url="http://api.test", token=token, username=username, seated=True
    )


# ------------------------------------------------------------------- manifests


def test_a_manifest_round_trips_without_a_secret():
    m = from_dict(manifest())
    d = to_dict(m)
    assert from_dict(d) == m.__class__(**m.__dict__)
    assert "token" not in json.dumps(d).lower().replace("token_url", "")


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"op": {"method": "DELETE"}}, "only GET"),
        ({"op": {"method": "POST"}}, "read_only_post"),
        ({"op": {"path": "https://evil.test/x"}}, "absolute on the base URL"),
        ({"op": {"path": "/a/../b"}}, "absolute on the base URL"),
        ({"op": {"path": "/users/{name}"}}, "undeclared name"),
        ({"headers": {"Host": "x"}}, "cannot be set"),
        ({"headers": {"Authorization": "x"}}, "carries the secret"),
        ({"auth": {"type": "query"}}, "names the query parameter"),
        ({"auth": {"type": "magic"}}, "auth.type"),
        ({"id": "Bad Id"}, "id must be"),
        ({"op": {"pagination": {"type": "cursor", "param": "after"}}}, "next cursor"),
        ({"base_url": "ftp://x"}, "http(s)"),
    ],
)
def test_a_manifest_that_could_do_harm_or_cannot_work_is_refused(change, reason):
    import re

    with pytest.raises(ManifestError, match=re.escape(reason.split()[0])):
        from_dict(manifest(**change))


def test_a_folder_of_manifests_loads_and_says_why_one_did_not(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRARY_CONNECTORS_DIR", str(tmp_path))
    registry.forget()
    (tmp_path / "good.json").write_text(json.dumps(manifest(id="good")))
    (tmp_path / "bad.json").write_text(json.dumps(manifest(id="bad", op={"method": "PUT"})))
    (tmp_path / "clash.json").write_text(json.dumps(manifest(id="sentinelone")))
    mods = registry.all_modules()
    assert "good" in mods and "sentinelone" in mods and "bad" not in mods
    errs = registry.errors()
    assert "only GET" in errs["bad.json"] and "already taken" in errs["clash.json"]
    registry.forget()


def test_configured_depends_on_the_auth():
    none = from_dict(manifest(auth={"type": "none"}))
    basic = from_dict(manifest(auth={"type": "basic"}))
    assert is_configured(none, ModuleConfig(id="m", base_url="http://x"))
    assert not is_configured(basic, ModuleConfig(id="m", base_url="http://x", token="pw"))
    assert is_configured(basic, ModuleConfig(id="m", base_url="http://x", token="pw", username="a"))


# ------------------------------------------------------------------- auth


async def test_bearer_and_a_nested_row_path(api):
    m = from_dict(manifest())
    res = await call(m, cfg(), m.op("items"), {})
    assert res["count"] == 2
    assert (
        render_rows(m.op("items"), res["rows"]).splitlines()[0] == "item 1 by u1 (2026-09-27 14:03)"
    )
    with pytest.raises(ModuleError, match="not authorised"):
        execute._cache.clear()
        await call(m, cfg(token="wrong"), m.op("items"), {})


async def test_key_in_the_query(api):
    m = from_dict(
        manifest(
            auth={"type": "query", "param": "api_key"},
            op={"path": "/query/items", "response": {"rows": "data", "line": "{title}"}},
        )
    )
    res = await call(m, cfg(token="k2"), m.op("items"), {})
    assert res["count"] == 1


async def test_basic_auth(api):
    m = from_dict(
        manifest(
            auth={"type": "basic"},
            op={"path": "/basic/items", "response": {"rows": "", "line": "{title}"}},
        )
    )
    res = await call(m, cfg(token="pw", username="alice"), m.op("items"), {})
    assert res["count"] == 3  # a top-level list is the rows


async def test_oauth_client_credentials_are_exchanged_once_and_cached(api):
    m = from_dict(
        manifest(
            auth={"type": "oauth2_client", "token_url": "/oauth/token"},
            op={"path": "/oauth/items", "response": {"rows": "", "line": "{title}"}},
        )
    )
    c = cfg(token="secret", username="client")
    assert (await call(m, c, m.op("items"), {}))["count"] == 1
    execute._cache.clear()
    assert (await call(m, c, m.op("items"), {}))["count"] == 1
    assert api.state.tokens == 1


# ------------------------------------------------------------------- parameters


async def test_path_and_enum_parameters_are_checked_and_encoded(api):
    m = from_dict(
        manifest(
            auth={"type": "none"},
            op={
                "path": "/repos/{owner}/{repo}/issues",
                "params": {
                    "owner": {"type": "string", "description": "owner"},
                    "repo": {"type": "string"},
                    "state": {"type": "enum", "values": ["open", "closed"], "default": "open"},
                },
                "response": {"rows": "", "line": "{title} ({state})"},
            },
        )
    )
    op = m.op("items")
    assert op.params["required"] == ["owner", "repo"]  # path parameters are required
    res = await call(m, cfg(), op, {"owner": "acme", "repo": "lib", "state": "closed"})
    assert render_rows(op, res["rows"]).splitlines()[0] == "acme/lib #0 (closed)"
    with pytest.raises(ModuleError, match="one of"):
        await call(m, cfg(), op, {"owner": "a", "repo": "b", "state": "merged"})
    with pytest.raises(ModuleError, match="required"):
        await call(m, cfg(), op, {"owner": "a"})
    path, _q, _b = execute.build(op, {"owner": "a/../b", "repo": "x?y=1"})
    assert path == "/repos/a%2F..%2Fb/x%3Fy%3D1/issues"  # a value cannot add a segment or query


async def test_a_read_only_search_posts_its_body(api):
    m = from_dict(
        manifest(
            auth={"type": "none"},
            op={
                "method": "POST",
                "read_only_post": True,
                "path": "/search",
                "params": {"q": {"type": "string", "in": "body", "required": True}},
                "body": {"query": {"match": {"title": "{q}"}}, "size": 5},
                "response": {"rows": "hits.hits[*]._source", "line": "{title}"},
            },
        )
    )
    res = await call(m, cfg(), m.op("items"), {"q": "raft"})
    assert api.state.body == {"query": {"match": {"title": "raft"}}, "size": 5}
    assert render_rows(m.op("items"), res["rows"]) == "hit for raft"


# ------------------------------------------------------------------- pagination


@pytest.mark.parametrize(
    ("path", "pagination", "pages"),
    [
        (
            "/paged",
            {"type": "page", "param": "page", "size_param": "per_page", "size": 3, "max_pages": 5},
            3,
        ),
        (
            "/offset",
            {
                "type": "offset",
                "param": "offset",
                "size_param": "limit",
                "size": 3,
                "start": 0,
                "max_pages": 5,
            },
            3,
        ),
        (
            "/cursor",
            {"type": "cursor", "param": "after", "cursor_path": "meta.next", "max_pages": 5},
            3,
        ),
        ("/linked", {"type": "link", "max_pages": 5}, 3),
    ],
)
async def test_every_kind_of_pagination_gathers_all_rows(api, path, pagination, pages):
    m = from_dict(
        manifest(
            auth={"type": "none"},
            op={
                "path": path,
                "pagination": pagination,
                "response": {"rows": "items", "line": "{title}", "limit": 50},
            },
        )
    )
    res = await call(m, cfg(), m.op("items"), {})
    assert res["count"] == 7 and res["pages"] == pages


async def test_pages_stop_at_the_cap(api):
    m = from_dict(
        manifest(
            auth={"type": "none"},
            op={
                "path": "/paged",
                "pagination": {
                    "type": "page",
                    "param": "page",
                    "size_param": "per_page",
                    "size": 3,
                    "max_pages": 2,
                },
                "response": {"rows": "items", "line": "{title}", "limit": 50},
            },
        )
    )
    res = await call(m, cfg(), m.op("items"), {})
    assert res["pages"] == 2 and res["count"] == 6


async def test_a_next_link_off_the_pinned_host_is_refused(api):
    m = from_dict(
        manifest(
            auth={"type": "none"},
            op={
                "path": "/linked-off",
                "pagination": {"type": "link"},
                "response": {"rows": "items", "line": "{title}", "limit": 50},
            },
        )
    )
    with pytest.raises(ModuleError, match="off the pinned host"):
        await call(m, cfg(), m.op("items"), {})


# ------------------------------------------------------------------- formats


async def test_xml_csv_and_text_responses(api):
    xml = from_dict(
        manifest(
            auth={"type": "none"},
            op={
                "path": "/feed.xml",
                "response": {"format": "xml", "rows": "channel/item", "line": "{title} <{link}>"},
            },
        )
    )
    res = await call(xml, cfg(), xml.op("items"), {})
    assert render_rows(xml.op("items"), res["rows"]) == "A <http://a>\nB <http://b>"
    csvm = from_dict(
        manifest(
            auth={"type": "none"},
            op={"path": "/table.csv", "response": {"format": "csv", "line": "{host} runs {os}"}},
        )
    )
    res = await call(csvm, cfg(), csvm.op("items"), {})
    assert render_rows(csvm.op("items"), res["rows"]) == "web-1 runs linux\ndb-1 runs bsd"
    txt = from_dict(
        manifest(
            auth={"type": "none"},
            op={"path": "/log.txt", "response": {"format": "text", "line": "{line}"}},
        )
    )
    res = await call(txt, cfg(), txt.op("items"), {})
    assert res["count"] == 2


async def test_xml_entities_are_refused(api):
    m = from_dict(
        manifest(
            auth={"type": "none"},
            op={"path": "/bomb.xml", "response": {"format": "xml", "rows": "", "line": "{text}"}},
        )
    )
    with pytest.raises(ModuleError, match="entities"):
        await call(m, cfg(), m.op("items"), {})


# ------------------------------------------------------------------- guard rails


async def test_a_response_over_the_byte_cap_is_refused(api):
    m = from_dict(
        manifest(
            auth={"type": "none"},
            limits={"max_bytes": 10_000},
            op={"path": "/big", "response": {"rows": "data", "line": "{value}"}},
        )
    )
    with pytest.raises(ModuleError, match="over 10,000 bytes"):
        await call(m, cfg(), m.op("items"), {})


async def test_the_rate_limit_holds(api):
    m = from_dict(
        manifest(
            auth={"type": "none"},
            limits={"rate_per_minute": 2, "cache_seconds": 0},
            op={"path": "/log.txt", "response": {"format": "text", "line": "{line}"}},
        )
    )
    await call(m, cfg(), m.op("items"), {})
    await call(m, cfg(), m.op("items"), {})
    with pytest.raises(ModuleError, match="rate limit"):
        await call(m, cfg(), m.op("items"), {})


# ------------------------------------------------------------------- rendering


def test_line_filters():
    row = {
        "t": "a long title that goes on and on",
        "labels": [{"name": "bug"}, {"name": "ui"}],
        "n": None,
        "at": 1790000000,
        "tags": ["x", "y"],
    }
    assert format_row("{t|trunc:12}", row) == "a long…"
    assert format_row("{labels|join} / {tags|join} / {tags|count}", row) == "bug, ui / x, y / 2"
    assert format_row("{n|default:none} {missing}", row) == "none"
    assert format_row("{at|date}", row).startswith("2026-")
    assert format_row("{labels[1].name|upper}", row) == "UI"
