"""Other providers: the file, the prefix routing, the OpenAI-protocol client against a
fake server, and the settings routes."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from library_agent.llm import openai_compat, providers
from library_agent.llm.client import LLM
from library_agent.llm.ollama import OllamaError


@pytest.fixture
def pfile(tmp_path, monkeypatch):
    p = tmp_path / "providers.json"
    monkeypatch.setenv("LIBRARY_PROVIDERS_FILE", str(p))
    providers._cache = None
    return p


# --- a provider that speaks the protocol ------------------------------------------------


def fake_server(*, json_schema_ok: bool = True, reasoning: bool = True) -> FastAPI:
    app = FastAPI()
    app.state.calls = []

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "big-model"}, {"id": "small-model"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        app.state.calls.append(body)
        if req.headers.get("authorization") != "Bearer sk-test-0000-test":
            return _err(401, "bad key")
        rf = body.get("response_format") or {}
        if rf.get("type") == "json_schema" and not json_schema_ok:
            return _err(400, "response_format.type json_schema unsupported")
        if body.get("stream"):

            async def gen():
                if reasoning:
                    yield "data: " + json.dumps(
                        {"choices": [{"delta": {"reasoning_content": "thinking... "}}]}
                    ) + "\n\n"
                for piece in ("Hello", ", ", "world"):
                    yield "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        if rf:
            text = json.dumps({"title": "A reply", "tags": ["x", "y"]})
        else:
            text = "ready"
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    return app


def _err(code: int, msg: str):
    from fastapi.responses import JSONResponse

    return JSONResponse({"error": {"message": msg}}, status_code=code)


@pytest.fixture
def served(monkeypatch):
    """Point every OpenAICompat client at an in-process fake, keeping the real headers."""
    holder: dict[str, FastAPI] = {}

    real = httpx.AsyncClient

    def client(*a, **kw):
        if "acme.test" in str(kw.get("base_url", "")):
            kw["transport"] = httpx.ASGITransport(app=holder["app"])
        return real(*a, **kw)

    monkeypatch.setattr(openai_compat.httpx, "AsyncClient", client)

    def use(app: FastAPI) -> FastAPI:
        holder["app"] = app
        return app

    return use


def _configure(pfile, key="sk-test-0000-test"):
    cfg = providers.Config()
    cfg.providers["acme"] = providers.Provider(
        id="acme", name="Acme", base_url="http://acme.test/v1", api_key=key
    )
    providers.save(cfg)
    return cfg


def test_file_round_trip_and_privacy(pfile):
    cfg = _configure(pfile)
    cfg.models["chat_general"] = "acme:big-model"
    providers.save(cfg)
    assert oct(pfile.stat().st_mode & 0o777) == "0o600"
    back = providers.load()
    assert back.providers["acme"].api_key == "sk-test-0000-test"
    assert back.models == {"chat_general": "acme:big-model"}
    pub = back.providers["acme"].public()
    assert "api_key" not in pub and pub["has_key"] and pub["key_tail"] == "test"


def test_prefix_routing_leaves_ollama_names_alone(pfile):
    _configure(pfile)
    assert providers.split("qwen3:30b-a3b") == (None, "qwen3:30b-a3b")
    prov, name = providers.split("acme:big-model")
    assert prov is not None and prov.id == "acme" and name == "big-model"
    assert providers.split("acme:") == (None, "acme:")
    assert providers.split("nobody:model") == (None, "nobody:model")


def test_roles_default_to_the_environment(pfile):
    from library_agent.config import settings

    assert providers.model_for("reading") == settings().reader_model
    assert providers.model_for("threads") == settings().reader_model
    assert providers.chat_options()["general"] == settings().chat_model_options["general"]
    cfg = _configure(pfile)
    cfg.models["threads"] = "acme:big-model"
    cfg.models["vision"] = ""
    providers.save(cfg)
    assert providers.model_for("threads") == "acme:big-model"
    assert providers.model_for("reading") == settings().reader_model
    assert providers.model_for("vision") == ""  # an override of nothing turns it off
    assert providers.is_remote("acme:big-model") and not providers.is_remote("qwen3:8b")


async def test_openai_client_generate_stream_structured(pfile, served):
    app = served(fake_server())
    _configure(pfile)
    async with LLM() as c:
        text = await c.generate("acme:big-model", "say ready")
        assert text == "ready"
        assert app.state.calls[-1]["model"] == "big-model"
        pieces = [p async for p in c.chat_stream("acme:big-model", [{"role": "user", "content": "hi"}])]
        assert ("thinking", "thinking... ") in pieces
        assert "".join(t for k, t in pieces if k == "content") == "Hello, world"
        out = await c.structured(
            "acme:big-model",
            "title it",
            {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
        )
        assert out["title"] == "A reply"
        assert app.state.calls[-1]["response_format"]["type"] == "json_schema"
        assert await c.supports_thinking("acme:big-model") is False


async def test_json_schema_fallback_is_learned(pfile, served):
    app = served(fake_server(json_schema_ok=False))
    _configure(pfile)
    openai_compat._JSON_SCHEMA_OK.clear()
    schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    async with LLM() as c:
        out = await c.structured("acme:big-model", "title it", schema)
        assert out["title"] == "A reply"
        # first ask json_schema, refused, then json_object with the schema in the prompt
        kinds = [x["response_format"]["type"] for x in app.state.calls]
        assert kinds == ["json_schema", "json_object"]
        assert "JSON schema" in app.state.calls[-1]["messages"][0]["content"]
        await c.structured("acme:big-model", "again", schema)
        assert app.state.calls[-1]["response_format"]["type"] == "json_object"
    assert openai_compat._JSON_SCHEMA_OK["acme"] is False


async def test_bad_key_is_a_clear_error(pfile, served):
    served(fake_server())
    _configure(pfile, key="sk-wrong")
    async with LLM() as c:
        with pytest.raises(OllamaError, match="HTTP 401"):
            await c.generate("acme:big-model", "hi")


async def test_probe_lists_models_and_answers(pfile, served):
    served(fake_server())
    cfg = _configure(pfile)
    async with openai_compat.OpenAICompat(cfg.providers["acme"]) as c:
        out = await c.probe()
    assert out["ok"] and out["models"] == ["big-model", "small-model"] and out["reply"] == "ready"


# --- the settings routes --------------------------------------------------------------


async def test_settings_routes(pfile, served, monkeypatch):
    served(fake_server())
    from library_agent.api.app import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        r = await c.put(
            "/api/settings/providers/acme",
            json={"name": "Acme", "base_url": "http://acme.test/v1/", "api_key": "sk-test-0000-test"},
        )
        assert r.status_code == 200 and r.json()["base_url"] == "http://acme.test/v1"
        assert "api_key" not in r.json()
        r = await c.put("/api/settings/providers/Bad Id", json={"base_url": "http://x"})
        assert r.status_code == 422
        r = await c.post("/api/settings/providers/acme/test")
        assert r.json()["ok"] and r.json()["reply"] == "ready"
        r = await c.put("/api/settings/models", json={"chat_general": "acme:big-model"})
        assert r.status_code == 200 and r.json()["chat_general"] == "acme:big-model"
        r = await c.get("/api/settings/providers")
        j = r.json()
        assert j["providers"][0]["id"] == "acme" and j["providers"][0]["has_key"]
        role = next(x for x in j["roles"] if x["id"] == "chat_general")
        assert role["overridden"] and role["remote"] and role["model"] == "acme:big-model"
        assert "acme:big-model" in j["catalogue"]["acme"]
        # keep the key when it is not sent again
        r = await c.put("/api/settings/providers/acme", json={"base_url": "http://acme.test/v1"})
        assert providers.load().providers["acme"].api_key == "sk-test-0000-test"
        r = await c.put("/api/settings/models", json={"chat_general": None})
        assert not providers.load().models
        r = await c.put("/api/settings/models", json={"threads": "acme:big-model"})
        r = await c.delete("/api/settings/providers/acme")
        assert r.json()["roles_reset"] == ["threads"]
        assert not providers.load().providers and not providers.load().models
