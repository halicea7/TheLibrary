"""Thin Ollama client. Structured output goes through the `format` JSON-schema parameter
rather than prompt-level pleading, which is what makes the Tier 1 passes reliable on a 4B."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any, Self

import httpx

from library_agent.config import settings


class OllamaError(RuntimeError):
    pass


# Models under a too-small context emit schema-shaped placeholders rather than failing,
# which is far worse than an error because it silently writes junk into the index.
_PLACEHOLDER = re.compile(r"^[.\u2026\s]*$|^(string|text|summary|todo|n/a)$", re.IGNORECASE)


def is_placeholder(value: object) -> bool:
    if isinstance(value, str):
        return bool(_PLACEHOLDER.match(value.strip()))
    if isinstance(value, list):
        return bool(value) and all(is_placeholder(v) for v in value)
    return False


def echoes_prompt(value: object, prompt: str, *, min_len: int = 30) -> bool:
    """A field whose value is lifted verbatim from the prompt is an instruction echo --
    the model copied the field's description instead of answering. Seen in the wild:
    explanation = "if they conflict, what each side claims and why they cannot both hold".
    """
    if not isinstance(value, str) or len(value.strip()) < min_len:
        return False
    return value.strip()[:120].lower() in prompt.lower()


def size_context(prompt_chars: int, *, reserve_tokens: int = 3072, floor: int = 8192) -> int:
    """Pick num_ctx from the prompt length.

    A fixed 8192 silently produced '...' for every field of a 1900-token document-summary
    prompt. Sizing to the input removes that whole failure mode."""
    needed = prompt_chars // 3 + reserve_tokens
    ctx = floor
    while ctx < needed and ctx < 65536:
        ctx *= 2
    return ctx


# Per model: does it support the `think` parameter? Sending think=true to a model that
# lacks it is a hard 400 ("does not support thinking"), which is exactly what happened
# when the non-thinking coder model was selected in chat. Cached for the process.
_THINKING: dict[str, bool] = {}


class Ollama:
    def __init__(self, base_url: str | None = None, timeout: float = 1800.0):
        cfg = settings()
        self._base = (base_url or cfg.ollama_url).rstrip("/")
        headers = {"Authorization": f"Bearer {cfg.ollama_api_key}"} if cfg.ollama_api_key else {}
        # Very generous. Ollama serialises requests against a single model, so a call can
        # sit queued behind another job's generation for a long time before it even
        # starts. A 600s timeout killed the contradiction pass twice while Tier 2 held
        # the model -- the request was fine, it just had not been reached yet.
        self._client = httpx.AsyncClient(base_url=self._base, timeout=timeout, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.3,
        num_ctx: int | None = None,
        think: bool = False,
        timeout: float = 300.0,
        num_predict: int = 1500,
        seed: int | None = None,
    ) -> str:
        """`timeout` is per request and deliberately much shorter than the client default.
        Ollama occasionally wedges on a single generation and never returns; it aborts
        that generation when the client disconnects, so a short timeout is what lets the
        next call through. A hung structured call is a lost item, not a lost hour."""
        num_ctx = num_ctx or size_context(len(prompt) + len(system or ""))
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": think,
            "keep_alive": settings().keep_alive,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        if system:
            payload["system"] = system
        if seed is not None:
            payload["options"]["seed"] = seed
        if schema:
            payload["format"] = schema
            # Grammar-constrained sampling can run away on a long free-text field and
            # never finish. Every structured output here fits comfortably in this; a
            # cap turns a hang into a truncated (and retried) response.
            payload["options"]["num_predict"] = num_predict
        r = await self._client.post("/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "")

    async def structured(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        num_ctx: int | None = None,
        retries: int = 2,
        instructions: str | None = None,
        num_predict: int = 1500,
        think: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate and parse JSON. Retries on malformed output — even with a schema,
        small models occasionally emit a stray prefix.

        `think=True` routes through /api/chat with thinking on, for the few calls that
        need judgement rather than extraction (organising the shelf). Thinking and a JSON
        schema do work together -- the earlier "returns empty" finding was the thinking
        eating the whole output budget -- so the budget is raised to cover both, and an
        empty answer is retried.

        `instructions` is the unfilled prompt template, if the caller wants echo
        protection: a field copied verbatim from the *instructions* is a failure and
        is retried. It must not be the filled prompt -- Tier 1 prompts embed the source
        text, and a summary that opens with a sentence from the abstract is fine."""
        last: Exception | None = None
        ctx = num_ctx or size_context(len(prompt) + len(system or ""))
        for _ in range(retries + 1):
            if think and await self.supports_thinking(model):
                raw = await self._generate_thinking(
                    model,
                    prompt,
                    system=system,
                    schema=schema,
                    temperature=temperature,
                    num_ctx=ctx,
                )
                if not raw.strip():
                    last = OllamaError("thinking consumed the output budget")
                    continue
            else:
                raw = await self.generate(
                    model,
                    prompt,
                    system=system,
                    schema=schema,
                    temperature=temperature,
                    num_ctx=ctx,
                    num_predict=num_predict,
                    seed=seed,
                )
            parsed: dict[str, Any] | None = None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                last = exc
                start, end = raw.find("{"), raw.rfind("}")
                if start != -1 and end > start:
                    try:
                        parsed = json.loads(raw[start : end + 1])
                    except json.JSONDecodeError:
                        parsed = None
            if parsed is not None:
                required = list(schema.get("required") or parsed.keys())
                if any(is_placeholder(parsed.get(k)) for k in required):
                    last = OllamaError("model returned placeholder values")
                    ctx = min(ctx * 2, 65536)  # almost always a context squeeze
                    continue
                if instructions and any(
                    echoes_prompt(parsed.get(k), instructions) for k in required
                ):
                    # Not a context problem; a sampling one. Nudge temperature so the
                    # retry does not reproduce the same echo.
                    last = OllamaError("model echoed the prompt into a field")
                    temperature = min(temperature + 0.2, 0.7)
                    continue
                return parsed
        raise OllamaError(f"model {model} returned unusable output: {last}")

    async def _generate_thinking(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None,
        schema: dict[str, Any],
        temperature: float,
        num_ctx: int,
        timeout: float = 600.0,
    ) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": True,
            "format": schema,
            "keep_alive": settings().keep_alive,
            # Room for the reasoning and the answer both; the grammar bounds the answer.
            "options": {
                "temperature": temperature,
                "num_ctx": max(num_ctx, 16384),
                "num_predict": 12000,
            },
        }
        r = await self._client.post("/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return (r.json().get("message") or {}).get("content", "")

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.6,
        num_ctx: int = 16384,
        seed: int | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yields ("thinking", text) and ("content", text) pieces. A `seed` with
        temperature 0 makes the same messages produce the same text.

        Thinking is left ON deliberately. Measured on qwen3:30b-a3b: with think=false the
        model still reasons, it just writes the reasoning into the visible answer ("Hmm,
        the user is asking me to..."), and Qwen's /no_think soft switch does not help on
        this build. Thinking on is the only way to get a clean answer -- it costs time
        before the first visible token, so the caller should surface it rather than
        leave the user staring at nothing. This was the true cause of every slow first
        token previously blamed on memory and prefill."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": settings().keep_alive,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        if seed is not None:
            payload["options"]["seed"] = seed
        # Only ask a model to think if it can; the others just answer.
        if await self.supports_thinking(model):
            payload["think"] = True
        # Long thinking is legitimate; endless is not.
        async with self._client.stream("POST", "/api/chat", json=payload, timeout=900.0) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                if piece := msg.get("thinking"):
                    yield "thinking", piece
                if piece := msg.get("content"):
                    yield "content", piece
                if chunk.get("done"):
                    return

    async def supports_thinking(self, model: str) -> bool:
        if model not in _THINKING:
            try:
                r = await self._client.post("/api/show", json={"model": model})
                r.raise_for_status()
                _THINKING[model] = "thinking" in (r.json().get("capabilities") or [])
            except Exception:  # noqa: BLE001 - assume no rather than fail the turn
                _THINKING[model] = False
        return _THINKING[model]

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        r = await self._client.post(
            "/api/embed",
            json={
                "model": model or settings().embed_model,
                "input": texts,
                "keep_alive": settings().keep_alive,
            },
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    async def loaded_models(self) -> list[str]:
        r = await self._client.get("/api/ps")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
