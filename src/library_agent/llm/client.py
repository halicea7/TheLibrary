"""The one client the library talks to.

An Ollama client that also knows the configured providers: a model named with a
provider's prefix goes to that provider over the OpenAI protocol, any other model goes
to Ollama exactly as before. Embeddings go to Ollama unless the settings name a provider
that serves the library's embedding model; the resident-model list is always Ollama's.
Backends are opened on first use and closed together."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from library_agent.config import settings
from library_agent.llm import providers
from library_agent.llm.ollama import Ollama
from library_agent.llm.openai_compat import OpenAICompat


class LLM(Ollama):
    def __init__(self, base_url: str | None = None, timeout: float = 1800.0):
        super().__init__(base_url, timeout)
        self._remote: dict[str, OpenAICompat] = {}

    def _route(self, model: str) -> tuple[OpenAICompat | None, str]:
        prov, name = providers.split(model)
        if prov is None:
            return None, model
        return self._client_for(prov), name

    def _client_for(self, prov: providers.Provider) -> OpenAICompat:
        # A provider edited in the settings gets a fresh client, since the key or URL
        # may have changed under us.
        cur = self._remote.get(prov.id)
        if cur is None or cur.provider != prov:
            if cur is not None:
                self._stale = getattr(self, "_stale", []) + [cur]
            cur = self._remote[prov.id] = OpenAICompat(prov)
        return cur

    async def aclose(self) -> None:
        for c in list(self._remote.values()) + list(getattr(self, "_stale", [])):
            await c.aclose()
        self._remote.clear()
        await super().aclose()

    async def generate(self, model: str, prompt: str, **kw: Any) -> str:
        remote, name = self._route(model)
        if remote:
            return await remote.generate(name, prompt, **kw)
        return await super().generate(model, prompt, **kw)

    async def structured(
        self, model: str, prompt: str, schema: dict[str, Any], **kw: Any
    ) -> dict[str, Any]:
        remote, name = self._route(model)
        if remote:
            return await remote.structured(name, prompt, schema, **kw)
        return await super().structured(model, prompt, schema, **kw)

    async def chat_stream(
        self, model: str, messages: list[dict[str, str]], **kw: Any
    ) -> AsyncIterator[tuple[str, str]]:
        remote, name = self._route(model)
        if remote:
            async for piece in remote.chat_stream(name, messages, **kw):
                yield piece
            return
        async for piece in super().chat_stream(model, messages, **kw):
            yield piece

    async def describe_image(self, model: str, prompt: str, png: bytes, **kw: Any) -> str:
        remote, name = self._route(model)
        if remote:
            return await remote.describe_image(name, prompt, png, **kw)
        return await super().describe_image(model, prompt, png, **kw)

    async def supports_thinking(self, model: str) -> bool:
        remote, name = self._route(model)
        if remote:
            return await remote.supports_thinking(name)
        return await super().supports_thinking(model)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        # Only the library's own embedding model moves; any other name asked for by hand
        # is an Ollama model, as before.
        if model is None or model == settings().embed_model:
            prov, name = providers.embed_route()
            if prov is not None:
                return await self._client_for(prov).embed(texts, name)
        return await super().embed(texts, model)
