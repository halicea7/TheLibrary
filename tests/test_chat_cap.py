"""Chat streaming asks for a finite answer: a model that loops terminates instead of
streaming until the request times out."""

from __future__ import annotations

import json

import pytest

from library_agent.llm.ollama import Ollama


class _Capture:
    """Stands in for the Ollama HTTP client, recording the payload and streaming back a
    two-chunk answer."""

    def __init__(self):
        self.payload = None

    def stream(self, method, path, **kw):
        self.payload = kw.get("json")
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    async def aclose(self):
        pass

    async def aiter_lines(self):
        yield json.dumps({"message": {"content": "hello"}})
        yield json.dumps({"message": {"content": " world"}, "done": True})


@pytest.mark.parametrize("thinks,expected", [(False, 4096), (True, 16384)])
async def test_chat_stream_caps_output(monkeypatch, thinks, expected):
    c = Ollama()
    cap = _Capture()
    c._client = cap  # type: ignore[assignment]

    async def fake_thinking(model):
        return thinks

    monkeypatch.setattr(c, "supports_thinking", fake_thinking)
    out = [piece async for piece in c.chat_stream("m", [{"role": "user", "content": "hi"}])]
    assert out == [("content", "hello"), ("content", " world")]
    assert cap.payload["options"]["num_predict"] == expected
    await c.aclose()
