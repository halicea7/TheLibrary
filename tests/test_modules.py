"""Modules: the safe GET executor, the JSON→passage rendering, and the consult step that
picks one operation and runs it — all against an in-process fake SentinelOne."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse

from library_agent.modules import execute
from library_agent.modules.builtin import SENTINELONE
from library_agent.modules.consult import consult
from library_agent.modules.execute import ModuleError, call
from library_agent.modules.render import render_rows
from library_agent.modules.store import ModuleConfig


def fake_s1() -> FastAPI:
    app = FastAPI()
    app.state.calls = []

    @app.get("/web/api/v2.1/application-management/risks")
    async def risks(
        request: Request,
        authorization: str | None = Header(default=None),
        cveId__contains: str | None = Query(default=None),
    ):
        app.state.calls.append(("risks", dict(request.query_params), authorization))
        if authorization != "ApiToken s1-secret":
            return JSONResponse({"errors": ["bad token"]}, status_code=401)
        rows = [
            {
                "cveId": "CVE-2024-3094",
                "applicationName": "xz",
                "endpointName": "HOST-1",
                "severity": "critical",
            }
        ]
        if cveId__contains and "3094" not in cveId__contains:
            rows = []
        return {"data": rows, "pagination": {"totalItems": len(rows)}}

    @app.get("/web/api/v2.1/threat-intelligence/iocs")
    async def iocs(authorization: str | None = Header(default=None)):
        return {
            "data": [
                {"value": "a" * 64, "name": "Known dropper", "type": "sha256", "source": "intel"}
            ]
        }

    @app.get("/leak")
    async def leak():  # a redirect off-host must never carry the token
        return RedirectResponse("http://evil.test/steal", status_code=302)

    return app


@pytest.fixture
def served(monkeypatch):
    app = fake_s1()
    real = httpx.AsyncClient

    def client(*a, **kw):
        if "s1.test" in str(kw.get("base_url", "")):
            kw["transport"] = httpx.ASGITransport(app=app)
        return real(*a, **kw)

    monkeypatch.setattr(execute.httpx, "AsyncClient", client)
    execute._cache.clear()
    return app


def _cfg(token="s1-secret"):
    return ModuleConfig(id="sentinelone", base_url="http://s1.test", token=token, seated=True)


async def test_get_is_built_from_the_manifest_and_authed(served):
    op = SENTINELONE.op("cve_exposure")
    res = await call(SENTINELONE, _cfg(), op, {"cve": "CVE-2024-3094"})
    assert res["ok"] and res["count"] == 1
    # the query key came from the manifest mapping, the const limit rode along, token attached
    _name, params, auth = served.state.calls[-1]
    assert params["cveId__contains"] == "CVE-2024-3094" and params["limit"] == "100"
    assert auth == "ApiToken s1-secret"
    assert "CVE-2024-3094: xz on HOST-1 — severity critical" in render_rows(op, res["rows"])


async def test_a_bad_token_is_a_clear_unauthorised(served):
    with pytest.raises(ModuleError, match="not authorised"):
        await call(
            SENTINELONE,
            _cfg(token="wrong"),
            SENTINELONE.op("cve_exposure"),
            {"cve": "CVE-2024-3094"},
        )


async def test_empty_result_renders_the_operations_own_line(served):
    op = SENTINELONE.op("cve_exposure")
    res = await call(SENTINELONE, _cfg(), op, {"cve": "CVE-2000-0000"})
    assert render_rows(op, res["rows"]) == op.render.empty


async def test_off_host_redirect_is_refused(served):
    # a bare operation pointed at the leaking path
    from library_agent.modules.manifest import Operation, Render

    leak = Operation(
        id="x",
        summary="",
        ask_when="",
        path="/leak",
        params={"type": "object"},
        query={},
        render=Render(rows="data", line="{v}"),
    )
    with pytest.raises(ModuleError, match="redirect off"):
        await call(SENTINELONE, _cfg(), leak, {})


class Picker:
    """A model that picks one operation and fills it from the question."""

    def __init__(self, operation, parameters):
        self.operation, self.parameters = operation, parameters

    async def structured(self, model, prompt, schema, **kw):
        assert "sentinelone.cve_exposure" in schema["properties"]["operation"]["enum"]
        return {"operation": self.operation, "parameters": self.parameters}


async def test_consult_picks_runs_and_renders(served):
    seated = [(SENTINELONE, _cfg())]
    c = Picker("sentinelone.cve_exposure", {"cve": "CVE-2024-3094"})
    out = await consult(c, "m", "who is exposed to CVE-2024-3094?", seated)
    assert len(out) == 1 and out[0].error == "" and out[0].count == 1
    assert "xz on HOST-1" in out[0].text
    assert out[0].label().startswith("SentinelOne · cve_exposure · CVE-2024-3094")


async def test_consult_none_is_the_common_case(served):
    out = await consult(
        Picker("none", {}), "m", "what is a race condition?", [(SENTINELONE, _cfg())]
    )
    assert out == []


async def test_consult_skips_when_a_required_param_is_missing(served):
    # the model chose an op but could not fill the required field: no call is made
    out = await consult(
        Picker("sentinelone.cve_exposure", {}), "m", "any exposure?", [(SENTINELONE, _cfg())]
    )
    assert out == [] and not served.state.calls


async def test_a_down_source_surfaces_as_an_error_not_silence(served, monkeypatch):
    async def boom(*a, **k):
        raise ModuleError("SentinelOne: ConnectError")

    monkeypatch.setattr("library_agent.modules.consult.call", boom)
    out = await consult(
        Picker("sentinelone.ioc_seen", {"indicator": "1.2.3.4"}),
        "m",
        "seen 1.2.3.4?",
        [(SENTINELONE, _cfg())],
    )
    assert len(out) == 1 and out[0].error and out[0].text == ""
