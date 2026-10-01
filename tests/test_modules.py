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
    """A model that makes one round of calls, then is done. `rounds` gives several rounds;
    each is a list of (operation, parameters)."""

    def __init__(self, operation=None, parameters=None, rounds=None):
        if rounds is None:
            rounds = [] if operation in (None, "none") else [[(operation, parameters or {})]]
        self.rounds, self.prompts = list(rounds), []

    async def structured(self, model, prompt, schema, **kw):
        enum = schema["properties"]["calls"]["items"]["properties"]["operation"]["enum"]
        assert "sentinelone.cve_exposure" in enum and kw.get("think") is True
        self.prompts.append(prompt)
        if not self.rounds:
            return {"calls": [], "done": True}
        calls = self.rounds.pop(0)
        return {
            "calls": [{"operation": o, "parameters": p} for o, p in calls],
            "done": not self.rounds,
        }


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


def test_an_unfillable_choice_falls_back_to_a_sibling_the_question_fills():
    # "Which hosts have Chrome?": the per-host lookup also needs the vendor, which the
    # question doesn't give; the inventory lookup takes just the name and still answers.
    from library_agent.modules.consult import _fillable

    ep, inv = SENTINELONE.op("app_endpoints"), SENTINELONE.op("app_inventory")
    assert _fillable(ep, {"application": "Chrome"}, SENTINELONE.operations) is inv
    assert (
        _fillable(ep, {"application": "Chrome", "vendor": "Google LLC"}, SENTINELONE.operations)
        is ep
    )
    assert _fillable(SENTINELONE.op("cve_exposure"), {}, SENTINELONE.operations) is None


def test_an_exact_input_is_never_taken_from_the_question():
    # "Google" in the question is not "Google LLC" in the inventory: an exact match would
    # find nothing and read as "not installed".
    assert SENTINELONE.op("app_endpoints").param_specs["vendor"]["exact"] is True
    from library_agent.modules.manifest import from_dict, to_dict

    back = from_dict(to_dict(SENTINELONE))
    assert back.op("app_endpoints").param_specs["vendor"]["exact"] is True


async def test_consult_drops_a_guessed_exact_input_and_falls_back(served, monkeypatch):
    from library_agent.modules import consult as cm

    ran = []

    async def fake_call(module, cfg, op, args):
        ran.append((op.id, args))
        return {"rows": [], "count": 0, "when": 0.0}

    monkeypatch.setattr(cm, "call", fake_call)
    c = Picker("sentinelone.app_endpoints", {"application": "Google Chrome", "vendor": "Google"})
    await consult(c, "m", "which hosts is Google Chrome installed on?", [(SENTINELONE, _cfg())])
    assert ran == [("app_inventory", {"application": "Google Chrome"})]


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


async def test_an_invented_value_that_breaks_the_inputs_pattern_is_not_used(served):
    # Measured: asked "What is Kerberoasting?", the model filled cve="Kerberoasting".
    c = Picker("sentinelone.cve_exposure", {"cve": "Kerberoasting"})
    out = await consult(c, "m", "What is Kerberoasting?", [(SENTINELONE, _cfg())])
    assert out == [] and not served.state.calls
    c = Picker("sentinelone.cve_exposure", {"cve": "cve-2024-3094"})  # any case
    out = await consult(c, "m", "who is exposed to cve-2024-3094?", [(SENTINELONE, _cfg())])
    assert len(out) == 1


def test_a_question_that_names_a_module_is_recognised():
    from library_agent.chat.session import _cannot_note, _names

    assert _names("Check Sentinel One and tell me the latest event", "SentinelOne")
    assert _names("anything on sentinelone?", "SentinelOne")
    assert not _names("what is a sentinel value?", "SentinelOne")
    note = _cannot_note(SENTINELONE)
    assert "not queried" in note and "recent" in note.lower()


async def test_several_calls_in_one_turn(served, monkeypatch):
    from library_agent.modules import consult as cm

    ran = []

    async def fake_call(module, cfg, op, args):
        ran.append((op.id, args))
        return {"rows": [], "count": 0, "when": 0.0}

    monkeypatch.setattr(cm, "call", fake_call)
    c = Picker(
        rounds=[
            [("sentinelone.agents", {}), ("sentinelone.app_inventory", {"application": "Zoom"})]
        ]
    )
    out = await consult(c, "m", "how many agents, and is Zoom installed?", [(SENTINELONE, _cfg())])
    assert sorted(r[0] for r in ran) == ["agents", "app_inventory"] and len(out) == 2


async def test_a_later_round_may_use_what_an_earlier_one_returned(served, monkeypatch):
    # The first call returns a CVE id; the second looks it up. The id was never in the
    # question -- it is grounded by the result.
    from library_agent.modules import consult as cm

    ran = []

    async def fake_call(module, cfg, op, args):
        ran.append((op.id, args))
        rows = [{"id": "CVE-2024-3094"}] if op.id == "recent_threats" else []
        return {"rows": rows, "count": len(rows), "when": 0.0}

    monkeypatch.setattr(cm, "call", fake_call)
    monkeypatch.setattr(cm, "render_rows", lambda op, rows: "\n".join(r["id"] for r in rows))
    c = Picker(
        rounds=[
            [("sentinelone.recent_threats", {})],
            [("sentinelone.cve_exposure", {"cve": "CVE-2024-3094"})],
        ]
    )
    await consult(c, "m", "what's the latest threat and who is exposed?", [(SENTINELONE, _cfg())])
    assert ran == [("recent_threats", {}), ("cve_exposure", {"cve": "CVE-2024-3094"})]
    assert "CVE-2024-3094" in c.prompts[1]  # the second round saw the first round's result


async def test_an_identifier_not_in_the_conversation_is_refused(served):
    # Measured: "look up those CVEs you just gave me" became CVE-2023-1234, which the
    # previous answer never mentioned.
    history = [
        ("user", "any recent Django CVEs?"),
        ("assistant", "CVE-2025-48432 and CVE-2025-57833."),
    ]
    c = Picker("sentinelone.cve_exposure", {"cve": "CVE-2023-1234"})
    out = await consult(c, "m", "look those up", [(SENTINELONE, _cfg())], history)
    assert out == [] and not served.state.calls
    c = Picker("sentinelone.cve_exposure", {"cve": "CVE-2025-48432"})
    out = await consult(c, "m", "look those up", [(SENTINELONE, _cfg())], history)
    assert len(out) == 1  # in the last answer: allowed


def test_a_rewrite_that_invents_an_identifier_is_not_used():
    from library_agent.chat.grounding import ungrounded

    known = "any recent Django CVEs? CVE-2025-48432 and CVE-2025-57833."
    assert ungrounded("look up CVE-2023-1234 and CVE-2023-5678 with NVD", known) == {
        "cve-2023-1234",
        "cve-2023-5678",
    }
    assert not ungrounded("look up CVE-2025-48432 with NVD", known)


def test_a_summary_line_counts_every_row_and_distinct_values():
    from library_agent.modules.render import summarise

    rows = [{"host": {"name": n}} for n in ("a", "b", "a", "c", "")]
    assert summarise("{count} rows on {distinct:host.name} hosts", rows) == "5 rows on 3 hosts"
    op = SENTINELONE.op("threats_matching")
    assert "{distinct:" in op.render.summary and op.params["required"] == ["text"]


def test_write_and_ask_gate_modules_by_every_model_that_reads_them(monkeypatch):
    from library_agent import classification as cls
    from library_agent.llm import providers
    from library_agent.modules import consult as cm
    from library_agent.modules import store

    monkeypatch.setattr(store, "seated_modules", lambda: [(SENTINELONE, _cfg())])
    monkeypatch.setattr(cls, "module_level", lambda mid, s=None: "internal")
    monkeypatch.setattr(providers, "is_trusted", lambda m: not m.startswith("cloud:"))
    s = cls.Scale()
    usable, held = cm.usable_modules(["local-model", "local-helper"], None, None, s)
    assert usable and not held
    # A checker on an untrusted provider reads the results too: local-only ops held back.
    usable, held = cm.usable_modules(["local-model", "cloud:helper"], None, None, s)
    assert held and "not marked internal" in held[0][2]
    # Classified above the ceiling: not consulted at all.
    usable, held = cm.usable_modules(["local-model"], ["public"], "public", s)
    assert not usable and "above this question's ceiling" in held[0][2]


def test_threats_can_be_found_by_host():
    op = SENTINELONE.op("threats_on_host")
    assert op.params["required"] == ["host"] and op.query["host"] == "computerName__contains"
    assert "{count}" in op.render.summary


def test_summaries_can_sum_a_field():
    from library_agent.modules.render import summarise

    rows = [{"n": 3}, {"n": "4"}, {"n": None}, {}]
    assert summarise("{count} apps, {sum:n} installs", rows) == "4 apps, 7 installs"


def test_sentinelone_counts_agents_by_os_and_reads_the_scan_schedule():
    by_os = SENTINELONE.op("agents_by_os")
    assert (
        by_os.const_query["countOnly"] == "true" and by_os.query["version"] == "osVersion__contains"
    )
    sched = SENTINELONE.op("inventory_scan_schedule")
    assert (
        sched.path.endswith("/application-management/settings") and "schedule" in sched.render.line
    )
    assert "{sum:endpointsCount}" in SENTINELONE.op("app_inventory").render.summary
