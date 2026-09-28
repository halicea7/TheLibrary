"""Tokens: a module seated in one of the bay's four sockets, and its design, checked."""

from __future__ import annotations

import base64
import io

import pytest
from fastapi import HTTPException

from library_agent.modules import store
from library_agent.modules.design import DesignError, clean, clean_mask


@pytest.fixture
def modfile(tmp_path, monkeypatch):
    p = tmp_path / "modules.json"
    monkeypatch.setenv("LIBRARY_MODULES_FILE", str(p))
    store._cache = None
    yield p
    store._cache = None


def _png(side=64, mode="RGB"):
    from PIL import Image

    img = Image.new(mode, (side, side), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def test_seating_takes_a_socket_and_refuses_a_taken_one(modfile, monkeypatch):
    from library_agent.api.routes import settings as routes
    from library_agent.modules.builtin import BUILTIN

    mid = next(iter(BUILTIN))
    out = await routes.put_module(mid, routes.ModulePut(socket=2))
    assert out["socket"] == 2 and out["seated"] is True
    # another module already in socket 2 is refused
    cfgs = store.load()
    cfgs["other"] = store.ModuleConfig(id="other", socket=1, seated=True)
    monkeypatch.setattr(store, "load", lambda: cfgs)
    with pytest.raises(HTTPException) as e:
        await routes.put_module(mid, routes.ModulePut(socket=1))
    assert e.value.status_code == 409
    out = await routes.put_module(mid, routes.ModulePut(socket=-1))
    assert out["socket"] is None and out["seated"] is False


async def test_the_old_checkbox_seats_in_the_first_free_socket(modfile):
    from library_agent.api.routes import settings as routes
    from library_agent.modules.builtin import BUILTIN

    mid = next(iter(BUILTIN))
    out = await routes.put_module(mid, routes.ModulePut(seated=True))
    assert out["socket"] == 0 and out["seated"]
    out = await routes.put_module(mid, routes.ModulePut(seated=False))
    assert out["socket"] is None and not out["seated"]


def test_a_file_from_before_sockets_keeps_its_seated_module(modfile):
    import json

    from library_agent.modules.builtin import BUILTIN

    mid = next(iter(BUILTIN))
    modfile.write_text(
        json.dumps({"modules": {mid: {"base_url": "https://x", "token": "t", "seated": True}}})
    )
    cfg = store.load()[mid]
    assert cfg.seated and cfg.socket == 0


def test_a_design_keeps_only_known_fields_of_the_right_kind():
    got = clean(
        {
            "material": "clear",
            "color": "#ABCDEF",
            "rimColor": "red",  # not a hex colour: dropped
            "tint": 7,  # clamped
            "planet": 3.9,
            "name": "ORBIT\x00 API" + "x" * 40,
            "rim": "yes",  # not a bool: dropped
            "script": "<b>",  # unknown: dropped
            "logoFinish": "neon",  # not a finish: dropped
        }
    )
    assert got == {
        "material": "clear",
        "color": "#abcdef",
        "tint": 1.0,
        "planet": 3,
        "name": ("ORBIT API" + "x" * 40)[:24],
    }


def test_the_logo_is_re_encoded_as_a_greyscale_png():
    out = clean_mask(_png(mode="RGB"))
    from PIL import Image

    raw = base64.b64decode(out.split(",", 1)[1])
    assert Image.open(io.BytesIO(raw)).mode == "L"
    assert clean_mask(None) is None and clean_mask("") is None
    with pytest.raises(DesignError):
        clean_mask("javascript:alert(1)")
    with pytest.raises(DesignError):
        clean_mask("data:image/png;base64," + base64.b64encode(b"not an image").decode())
    with pytest.raises(DesignError):
        clean_mask(_png(side=2000))
