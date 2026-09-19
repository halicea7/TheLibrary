"""Batched embedding. bge-m3 at 1024 dims; batches keep the GPU busy without
starving an interactive chat of headroom."""

from __future__ import annotations

from library_agent.config import settings
from library_agent.llm.ollama import Ollama

BATCH = 32


async def embed_texts(texts: list[str], client: Ollama | None = None) -> list[list[float]]:
    if not texts:
        return []
    own = client is None
    c = client or Ollama()
    try:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            out.extend(await c.embed(texts[i : i + BATCH], model=settings().embed_model))
        return out
    finally:
        if own:
            await c.aclose()


async def embed_query(text: str, client: Ollama | None = None) -> list[float]:
    return (await embed_texts([text], client))[0]
