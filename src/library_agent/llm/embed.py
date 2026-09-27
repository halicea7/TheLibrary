"""Batched embedding. bge-m3 at 1024 dims; batches keep the GPU busy without
starving an interactive chat of headroom."""

from __future__ import annotations

from library_agent.config import settings
from library_agent.llm.client import LLM
from library_agent.llm.ollama import Ollama

BATCH = 32


async def embed_texts(texts: list[str], client: Ollama | None = None) -> list[list[float]]:
    if not texts:
        return []
    own = client is None
    c = client or LLM()
    try:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            out.extend(await c.embed(texts[i : i + BATCH], model=settings().embed_model))
        return out
    finally:
        if own:
            await c.aclose()


# Query vectors, remembered briefly. On a machine where the embedder and a large writer
# cannot both stay loaded, each embedding call can evict the writer; a long document
# embeds all its searches in one batch up front (`prime_queries`) and every later lookup
# is served from here, so the writer is swapped out once, not once per section.
_QUERY_CACHE: dict[tuple[str, str], list[float]] = {}
_QUERY_CACHE_MAX = 1024


async def prime_queries(texts: list[str], client: Ollama | None = None) -> None:
    model = settings().embed_model
    todo = list(dict.fromkeys(t for t in texts if (model, t) not in _QUERY_CACHE))
    for t, v in zip(todo, await embed_texts(todo, client), strict=True):
        _remember(model, t, v)


def _remember(model: str, text: str, vec: list[float]) -> None:
    if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
        _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
    _QUERY_CACHE[(model, text)] = vec


async def embed_query(text: str, client: Ollama | None = None) -> list[float]:
    model = settings().embed_model
    hit = _QUERY_CACHE.get((model, text))
    if hit is not None:
        return hit
    vec = (await embed_texts([text], client))[0]
    _remember(model, text, vec)
    return vec
