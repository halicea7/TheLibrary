"""Is the model actually answering, and who gets to ask it right now.

Two things a shared library needs that a solo one did not.

**Liveness.** Ollama can wedge with its HTTP still up: `/api/tags` answers, both models
show resident, and no generation ever returns. A reachability check calls that healthy.
So a probe asks for one token every so often with a short timeout, and the answer to
"is the model answering" is that probe's last result. When it says no, the API refuses
new generations at once with a 503 and the reason, instead of holding every caller's
socket open for the full timeout.

On a shared GPU (several people's queries on the same box) one slow probe is load, not a
wedge: a probe that times out or breaks off marks the model *strained* -- requests still go
through, the desk says it is slow, and the next probe comes sooner -- and only several in a
row mark it down. A refused connection (nothing listening: the tunnel is down) is down at
once, since every request would fail the same way.

**The gate.** One Ollama serialises generations; without a gate a script in a loop puts
everyone behind it invisibly. The gate holds a small global limit and one in-flight
generation per client, and a caller that would wait longer than the queue timeout gets a
429 with Retry-After rather than a silent wait."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from library_agent.config import settings
from library_agent.llm import providers

log = logging.getLogger(__name__)


@dataclass
class Liveness:
    alive: bool = True
    # Slow or breaking off, but not (yet) down: requests go through; the desk says so.
    strained: bool = False
    failures: int = 0  # slow or broken probes in a row
    checked_at: float = 0.0
    latency: float | None = None
    detail: str = ""
    _task: asyncio.Task | None = field(default=None, repr=False)

    async def probe(self, timeout: float | None = None) -> bool:
        cfg = settings()
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(
                base_url=cfg.ollama_url, timeout=timeout or cfg.liveness_timeout_seconds
            ) as c:
                r = await c.post(
                    "/api/generate",
                    json={
                        "model": providers.ollama_probe_model(),
                        "prompt": "Reply with the single word: ready",
                        "stream": False,
                        "think": False,
                        "keep_alive": cfg.keep_alive,
                        "options": {"num_ctx": 512, "num_predict": 4, "temperature": 0},
                    },
                )
                r.raise_for_status()
                self.alive, self.strained, self.failures, self.detail = True, False, 0, ""
        except httpx.ConnectError as exc:
            # Nothing listening: down now, whatever came before.
            self.alive, self.strained = False, False
            self.failures = max(self.failures + 1, cfg.liveness_failures_to_down)
            self.detail = (
                f"cannot connect ({str(exc)[:80] or 'connection refused'}) -- is the tunnel up?"
            )
        except httpx.TimeoutException:
            self._fault(
                f"the model did not answer a one-token probe within "
                f"{timeout or cfg.liveness_timeout_seconds:.0f}s"
            )
        except Exception as exc:  # noqa: BLE001
            self._fault(f"{type(exc).__name__}: {str(exc)[:120]}")
        self.latency = round(time.monotonic() - t0, 2)
        self.checked_at = time.time()
        return self.alive

    def _fault(self, detail: str) -> None:
        """A slow or broken probe: strained until it happens enough times in a row."""
        self.failures += 1
        self.detail = detail
        if self.failures >= settings().liveness_failures_to_down:
            self.alive, self.strained = False, False
        else:
            self.alive, self.strained = True, True

    async def _loop(self) -> None:
        cfg = settings()
        while True:
            try:
                was, was_strained = self.alive, self.strained
                await self.probe()
                if was and not self.alive:
                    log.error("model liveness lost: %s", self.detail)
                elif not was and self.alive:
                    log.info("model liveness back after %.1fs", self.latency or 0)
                elif self.strained and not was_strained:
                    log.warning(
                        "model strained (%d/%d): %s",
                        self.failures,
                        cfg.liveness_failures_to_down,
                        self.detail,
                    )
            except Exception:
                log.debug("liveness loop error", exc_info=True)
            # Healthy: the usual interval. Strained or down: look again sooner.
            await asyncio.sleep(
                cfg.liveness_interval_seconds
                if self.alive and not self.strained
                else max(15, cfg.liveness_interval_seconds // 4)
            )

    def start(self) -> None:
        if not self._task:
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    def snapshot(self) -> dict:
        return {
            "alive": self.alive,
            "strained": self.strained,
            "failures": self.failures,
            "checked_seconds_ago": round(time.time() - self.checked_at)
            if self.checked_at
            else None,
            "latency": self.latency,
            "detail": self.detail,
        }


liveness = Liveness()


class Busy(Exception):
    def __init__(self, waiting: int, retry_after: int):
        super().__init__(f"the model is busy; {waiting} ahead of you")
        self.waiting = waiting
        self.retry_after = retry_after


class Gate:
    """Global concurrency plus one in-flight generation per client."""

    def __init__(self) -> None:
        self._global: asyncio.Semaphore | None = None
        self._clients: dict[str, int] = {}
        self._waiting = 0

    def _sem(self) -> asyncio.Semaphore:
        if self._global is None:
            self._global = asyncio.Semaphore(settings().max_concurrent_generations)
        return self._global

    async def acquire(self, client: str) -> None:
        cfg = settings()
        # The person at the desk is not rate-limited against themselves; tokens are.
        if client != "ui" and self._clients.get(client, 0) >= cfg.max_generations_per_client:
            raise Busy(self._waiting, 5)
        self._waiting += 1
        try:
            try:
                await asyncio.wait_for(self._sem().acquire(), timeout=cfg.queue_timeout_seconds)
            except TimeoutError as exc:
                raise Busy(self._waiting - 1, cfg.queue_timeout_seconds) from exc
        finally:
            self._waiting -= 1
        self._clients[client] = self._clients.get(client, 0) + 1

    def release(self, client: str) -> None:
        self._clients[client] = max(0, self._clients.get(client, 0) - 1)
        self._sem().release()

    def snapshot(self) -> dict:
        return {"in_flight": sum(self._clients.values()), "waiting": self._waiting}


gate = Gate()
