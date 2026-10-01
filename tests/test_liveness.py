"""The gate and the probe: what a shared library needs to stay predictable."""

from __future__ import annotations

import asyncio

import pytest

from library_agent.config import settings
from library_agent.llm.liveness import Busy, Gate, Liveness


class TestGate:
    async def test_one_in_flight_per_token_but_not_for_the_desk(self, monkeypatch):
        monkeypatch.setattr(settings(), "max_concurrent_generations", 4)
        monkeypatch.setattr(settings(), "max_generations_per_client", 1)
        g = Gate()
        await g.acquire("token:a")
        with pytest.raises(Busy):
            await g.acquire("token:a")
        await g.acquire("token:b")  # a different client is fine
        await g.acquire("ui")
        await g.acquire("ui")  # the person at the desk is not limited against themselves
        assert g.snapshot()["in_flight"] == 4
        g.release("token:a")
        await g.acquire("token:a")

    async def test_global_limit_times_out_into_busy(self, monkeypatch):
        monkeypatch.setattr(settings(), "max_concurrent_generations", 1)
        monkeypatch.setattr(settings(), "max_generations_per_client", 5)
        monkeypatch.setattr(settings(), "queue_timeout_seconds", 0.2)
        g = Gate()
        await g.acquire("token:a")
        with pytest.raises(Busy) as e:
            await g.acquire("token:b")
        assert e.value.retry_after >= 0
        g.release("token:a")
        await g.acquire("token:b")


class TestProbe:
    async def test_unreachable_model_is_not_alive(self, monkeypatch):
        monkeypatch.setattr(settings(), "ollama_url", "http://127.0.0.1:9")  # nothing listens
        lv = Liveness()
        assert await lv.probe(timeout=1.0) is False
        s = lv.snapshot()
        assert s["alive"] is False and s["detail"] and s["latency"] is not None

    async def test_a_hang_strains_and_only_repeated_hangs_mean_down(self, monkeypatch):
        # A server that accepts and never answers: a wedge, or a GPU busy with others' work.
        async def handler(reader, writer):
            await asyncio.sleep(5)
            writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr(settings(), "ollama_url", f"http://127.0.0.1:{port}")
        monkeypatch.setattr(settings(), "liveness_failures_to_down", 3)
        lv = Liveness()
        try:
            assert await lv.probe(timeout=0.3) is True  # once: busy, still serving
            assert lv.strained and lv.failures == 1 and "did not answer" in lv.detail
            assert await lv.probe(timeout=0.3) is True and lv.failures == 2
            assert await lv.probe(timeout=0.3) is False  # three in a row: down
            assert not lv.strained and lv.snapshot()["alive"] is False
        finally:
            server.close()

    async def test_nothing_listening_is_down_at_once(self, monkeypatch):
        monkeypatch.setattr(settings(), "ollama_url", "http://127.0.0.1:9")
        lv = Liveness()
        assert await lv.probe(timeout=1.0) is False and "tunnel" in lv.detail
