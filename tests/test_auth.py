"""Who may talk to the library: loopback is the UI and needs nothing; anyone else needs a
token, and there must be tokens for anyone else to be let in at all."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from library_agent.api.auth import BearerOrLoopback, token_ok
from library_agent.config import settings


def _app():
    app = FastAPI()
    app.add_middleware(BearerOrLoopback)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


class TestDoor:
    def test_loopback_needs_nothing(self):
        # TestClient's default client address is not loopback; set it explicitly.
        c = TestClient(_app(), client=("127.0.0.1", 1234))
        assert c.get("/ping").status_code == 200

    def test_off_box_refused_without_tokens(self, monkeypatch):
        monkeypatch.setattr(settings(), "api_tokens", "")
        c = TestClient(_app(), client=("10.0.0.7", 1234))
        r = c.get("/ping")
        assert r.status_code == 403 and "not open" in r.json()["detail"]

    def test_off_box_needs_a_valid_token(self, monkeypatch):
        monkeypatch.setattr(settings(), "api_tokens", "alpha-token, beta-token")
        c = TestClient(_app(), client=("10.0.0.7", 1234))
        assert c.get("/ping").status_code == 401
        assert c.get("/ping", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/ping", headers={"Authorization": "Bearer beta-token"}).status_code == 200

    def test_a_trusted_network_is_treated_as_this_machine(self, monkeypatch):
        # In a container the host's browser arrives from the network's gateway.
        monkeypatch.setattr(settings(), "api_tokens", "")
        monkeypatch.setattr(settings(), "trusted_networks", "172.31.77.1/32, 192.168.65.1/32")
        assert TestClient(_app(), client=("172.31.77.1", 1)).get("/ping").status_code == 200
        assert TestClient(_app(), client=("192.168.65.1", 1)).get("/ping").status_code == 200
        # Another container on the same network is not the gateway.
        assert TestClient(_app(), client=("172.31.77.5", 1)).get("/ping").status_code == 403
        monkeypatch.setattr(settings(), "trusted_networks", "")
        assert TestClient(_app(), client=("172.31.77.1", 1)).get("/ping").status_code == 403

    def test_token_check(self, monkeypatch):
        monkeypatch.setattr(settings(), "api_tokens", "s3cret")
        assert token_ok("Bearer s3cret") and token_ok("bearer s3cret")
        assert not token_ok("s3cret") and not token_ok(None) and not token_ok("Bearer S3CRET")
