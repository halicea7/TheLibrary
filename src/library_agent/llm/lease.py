"""Admission control between interactive chat and background reading.

All four models stay resident (~23GB against a ~36GB Metal budget), so this is purely
about latency, not memory — there is no model eviction to orchestrate. Chat marks itself
active; reading checks between work units and yields. Reading only resumes after a quiet
period, so a conversation doesn't thrash a job to a halt one message at a time."""

from __future__ import annotations

import time

from redis.asyncio import Redis

from library_agent.config import settings

CHAT_ACTIVE_KEY = "library:llm:chat_active"
LAST_CHAT_KEY = "library:llm:last_chat_ts"


def redis_client() -> Redis:
    return Redis.from_url(settings().redis_url, decode_responses=True)


async def mark_chat_active(r: Redis) -> None:
    """Called at the start of every chat turn and refreshed while streaming."""
    s = settings()
    async with r.pipeline(transaction=True) as pipe:
        pipe.set(CHAT_ACTIVE_KEY, "1", ex=s.chat_lease_ttl_seconds)
        pipe.set(LAST_CHAT_KEY, str(time.time()))
        await pipe.execute()


async def mark_chat_done(r: Redis) -> None:
    """Drops the active flag but leaves the timestamp, so the quiet period starts now."""
    async with r.pipeline(transaction=True) as pipe:
        pipe.delete(CHAT_ACTIVE_KEY)
        pipe.set(LAST_CHAT_KEY, str(time.time()))
        await pipe.execute()


async def reading_may_proceed(r: Redis) -> tuple[bool, str | None]:
    """Checked by the worker between units of work, never mid-call."""
    if await r.exists(CHAT_ACTIVE_KEY):
        return False, "chat_active"
    last = await r.get(LAST_CHAT_KEY)
    if last:
        quiet_for = time.time() - float(last)
        if quiet_for < settings().reading_resume_quiet_seconds:
            return False, f"chat_quiet_period ({quiet_for:.0f}s elapsed)"
    return True, None
