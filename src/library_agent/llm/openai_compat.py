"""A model behind the OpenAI chat-completions protocol.

The same surface as the Ollama client -- generate, structured, chat_stream,
describe_image -- so the router can hand a call to either without the caller knowing.
Structured output asks for `response_format: json_schema` first and, where the server
refuses that, falls back to `json_object` with the schema written into the system
prompt; the reply is judged by the same placeholder and echo guards either way, since
the failures are the model's, not the transport's. Thinking is the provider's to decide:
whatever reasoning a stream carries (`reasoning_content` or `reasoning`) is passed on as
the murmur, and nothing is asked for."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from collections.abc import AsyncIterator
from typing import Any, Self

import httpx

from library_agent.llm.ollama import OllamaError, clean, parse_structured

# Reasoning inline. Some servers (vLLM and Ollama's /v1 behind a router, among them)
# return a thinking model's reasoning inside `content`, closed by </think> -- often with
# no opening tag, since the chat template put that in the prompt -- instead of in
# `reasoning_content`. Left alone it is shown and saved as the answer, and it sits in
# front of the JSON of every structured reply.
_CLOSE, _OPEN = "</think>", "<think>"
_THINKING_FAMILY = re.compile(r"qwen3(?!-coder)|deepseek-r1|qwq|thinking|reasoner", re.IGNORECASE)
# Models seen reasoning inline, so their next stream is read that way from its first token.
_INLINE_THINKERS: set[str] = set()


def split_reasoning(text: str) -> tuple[str, str]:
    """(reasoning, answer) from a reply that may carry its reasoning inline."""
    if _CLOSE in text:
        head, _, tail = text.partition(_CLOSE)
        return head.replace(_OPEN, "", 1).strip(), tail.lstrip()
    if text.lstrip().startswith(_OPEN):  # opened and never closed: all reasoning
        return text.replace(_OPEN, "", 1).strip(), ""
    return "", text


def _expects_inline(provider_id: str, model: str) -> bool:
    return f"{provider_id}:{model}" in _INLINE_THINKERS or bool(_THINKING_FAMILY.search(model))


from library_agent.llm.providers import Provider

# Per provider: does it accept response_format=json_schema? Learned from the first 4xx.
_JSON_SCHEMA_OK: dict[str, bool] = {}


# Providers found to drop all but the first text of an embedding batch, by provider id.
_BATCH_EMBED_OK: dict[str, bool] = {}
EMBED_SINGLE_CONCURRENCY = 8


class OpenAICompat:
    def __init__(self, provider: Provider, timeout: float = 600.0):
        self.provider = provider
        headers = {"Content-Type": "application/json", **provider.headers}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        self._client = httpx.AsyncClient(
            base_url=provider.base_url, timeout=timeout, headers=headers
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    # -- the protocol ------------------------------------------------------------------

    async def models(self) -> list[str]:
        r = await self._client.get("/models", timeout=20)
        r.raise_for_status()
        data = r.json().get("data") or []
        return sorted(str(m.get("id")) for m in data if m.get("id"))

    async def _complete(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return await self._post("/chat/completions", payload, timeout)

    async def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        # A router in front of many machines rate-limits; a 429 or a passing 5xx is a
        # reason to wait, not to fail the pass. Retry-After is honoured when given.
        delay = 2.0
        for attempt in range(5):
            r = await self._client.post(path, json=payload, timeout=timeout)
            if r.status_code in (429, 502, 503, 504) and attempt < 4:
                wait = r.headers.get("retry-after")
                try:
                    pause = float(wait) if wait else delay
                except ValueError:
                    pause = delay
                await asyncio.sleep(min(pause, 60.0))
                delay = min(delay * 2, 30.0)
                continue
            if r.status_code >= 400:
                raise OllamaError(f"{self.provider.name}: HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        raise OllamaError(f"{self.provider.name}: still throttled after retries")

    @staticmethod
    def _text(reply: dict[str, Any]) -> str:
        choices = reply.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return clean(split_reasoning(content or "")[1])

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
        del num_ctx, think  # the provider sizes its own context and decides its own thinking
        messages = self._messages(system, prompt, schema if not self._json_schema_ok() else None)
        payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if seed is not None:
            payload["seed"] = seed
        if schema:
            payload["max_tokens"] = num_predict
            if self._json_schema_ok():
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": schema},
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        try:
            reply = await self._complete(payload, timeout)
        except OllamaError as exc:
            # A server that does not know json_schema says so with a 4xx; learn it and
            # ask the plainer way from now on.
            if (
                schema
                and self._json_schema_ok()
                and ("HTTP 400" in str(exc) or "HTTP 422" in str(exc))
            ):
                _JSON_SCHEMA_OK[self.provider.id] = False
                return await self.generate(
                    model,
                    prompt,
                    system=system,
                    schema=schema,
                    temperature=temperature,
                    timeout=timeout,
                    num_predict=num_predict,
                    seed=seed,
                )
            raise
        return self._text(reply)

    def _json_schema_ok(self) -> bool:
        return _JSON_SCHEMA_OK.get(self.provider.id, True)

    @staticmethod
    def _messages(
        system: str | None, prompt: str, inline_schema: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        sys_text = system or ""
        if inline_schema:
            sys_text = (sys_text + "\n\n" if sys_text else "") + (
                "Answer with a single JSON object and nothing else, matching this JSON schema:\n"
                + json.dumps(inline_schema)
            )
        msgs: list[dict[str, Any]] = []
        if sys_text:
            msgs.append({"role": "system", "content": sys_text})
        msgs.append({"role": "user", "content": prompt})
        return msgs

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
        last: Exception | None = None
        for _ in range(retries + 1):
            raw = await self.generate(
                model,
                prompt,
                system=system,
                schema=schema,
                temperature=temperature,
                num_ctx=num_ctx,
                think=think,
                num_predict=num_predict,
                seed=seed,
            )
            parsed, why = parse_structured(raw, schema, instructions)
            if parsed is not None:
                return parsed
            if why == "json":
                last = OllamaError("model returned malformed JSON")
            elif why == "placeholder":
                last = OllamaError("model returned placeholder values")
                num_predict = min(num_predict * 2, 8000)
            else:
                last = OllamaError("model echoed the prompt into a field")
                temperature = min(temperature + 0.2, 0.7)
        raise OllamaError(f"model {self.provider.id}:{model} returned unusable output: {last}")

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.6,
        num_ctx: int = 16384,
        seed: int | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        del num_ctx
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if seed is not None:
            payload["seed"] = seed
        key = f"{self.provider.id}:{model}"
        inline = _expects_inline(self.provider.id, model)
        held, shown, separate, thought = "", [], False, False
        async with self._client.stream(
            "POST", "/chat/completions", json=payload, timeout=900.0
        ) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode(errors="replace")
                raise OllamaError(f"{self.provider.name}: HTTP {r.status_code}: {body[:300]}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if piece := (delta.get("reasoning_content") or delta.get("reasoning")):
                        separate = True
                        yield "thinking", clean(piece)
                    if piece := delta.get("content"):
                        if not inline:
                            if _CLOSE in piece:  # reasoning inline after all: learn it
                                _INLINE_THINKERS.add(key)
                                before, _, after = piece.partition(_CLOSE)
                                if before:
                                    yield "content", clean(before)
                                if after.strip():
                                    yield "content", clean(after.lstrip())
                                continue
                            yield "content", clean(piece)
                            continue
                        # Reasoning until </think>: shown as thinking as it comes, with a
                        # tail held back in case the tag is split across two chunks.
                        held += piece
                        if _CLOSE in held:
                            before, _, after = held.partition(_CLOSE)
                            before = before.replace(_OPEN, "", 1)
                            if before:
                                yield "thinking", clean(before)
                            thought = True
                            inline, held = False, ""
                            if after.lstrip():
                                yield "content", clean(after.lstrip())
                            continue
                        if not separate and not thought and len(held) > len(_CLOSE):
                            out, held = held[: -len(_CLOSE)], held[-len(_CLOSE) :]
                            shown.append(out)
                            yield "thinking", clean(out.replace(_OPEN, "", 1))
                        elif separate:  # the server does split reasoning: this is answer
                            inline = False
                            yield "content", clean(held)
                            held = ""
        # The stream ended while still waiting for </think>: the model did not reason
        # this time, and what looked like reasoning was the answer.
        if inline and (shown or held):
            yield "content", clean("".join(shown) + held)

    async def describe_image(
        self,
        model: str,
        prompt: str,
        png: bytes,
        *,
        temperature: float = 0.2,
        num_predict: int = 260,
        timeout: float = 180.0,
    ) -> str:
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": num_predict,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
        }
        return self._text(await self._complete(payload, timeout))

    async def embed(
        self, texts: list[str], model: str, timeout: float = 120.0
    ) -> list[list[float]]:
        """The OpenAI embeddings call, in the order given (a server may answer out of order).
        A server that answers a batch with fewer vectors than texts -- one router embedded
        only the first -- is learned and sent one text per request from then on."""
        if not texts:
            return []
        if len(texts) > 1 and not _BATCH_EMBED_OK.get(self.provider.id, True):
            return await self._embed_singly(texts, model, timeout)
        reply = await self._post("/embeddings", {"model": model, "input": texts}, timeout)
        data = sorted(reply.get("data") or [], key=lambda d: d.get("index", 0))
        if len(data) == len(texts):
            return [d["embedding"] for d in data]
        if len(texts) > 1 and len(data) < len(texts):
            _BATCH_EMBED_OK[self.provider.id] = False
            return await self._embed_singly(texts, model, timeout)
        raise OllamaError(f"{self.provider.name}: {len(data)} embeddings for {len(texts)} texts")

    async def _embed_singly(
        self, texts: list[str], model: str, timeout: float
    ) -> list[list[float]]:
        sem = asyncio.Semaphore(EMBED_SINGLE_CONCURRENCY)

        async def one(t: str) -> list[float]:
            async with sem:
                reply = await self._post("/embeddings", {"model": model, "input": [t]}, timeout)
            data = reply.get("data") or []
            if len(data) != 1:
                raise OllamaError(f"{self.provider.name}: {len(data)} embeddings for 1 text")
            return data[0]["embedding"]

        return list(await asyncio.gather(*(one(t) for t in texts)))

    async def supports_thinking(self, model: str) -> bool:
        # Unknown until it streams; the reasoning is shown if it arrives, and the UI's
        # toggle is show/hide either way.
        return False

    async def probe(self, model: str | None = None) -> dict[str, Any]:
        """For the settings page: can we list models, and does one answer?"""
        out: dict[str, Any] = {"ok": False, "models": [], "reply": "", "error": ""}
        try:
            out["models"] = await self.models()
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"listing models: {type(exc).__name__}: {str(exc)[:200]}"
        # Ask one model, or the first few that look like they talk: an embedding or
        # reranking model listed first would fail the test for nothing.
        skip = ("embed", "bge", "rerank", "whisper", "tts")
        chatty = [m for m in out["models"] if not any(w in m.lower() for w in skip)]
        picks = [model] if model else (chatty or out["models"])[:3]
        for pick in picks:
            try:
                reply = await self._complete(
                    {
                        "model": pick,
                        "messages": [
                            {"role": "user", "content": "Reply with the single word: ready"}
                        ],
                        "max_tokens": 8,
                        "temperature": 0,
                    },
                    timeout=60,
                )
                out["reply"] = self._text(reply).strip()
                out["model"] = pick
                out["ok"] = True
                out["error"] = ""
                break
            except Exception as exc:  # noqa: BLE001
                out["error"] = f"{pick}: {type(exc).__name__}: {str(exc)[:300]}"
        return out
