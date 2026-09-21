"""End-of-Phase-0 verification: every external dependency reachable, models resident,
and a vector round-trips through Postgres."""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from library_agent.config import settings
from library_agent.db.session import session_scope
from library_agent.llm.client import LLM
from library_agent.llm.embed import embed_query
from library_agent.llm.lease import reading_may_proceed, redis_client


async def main() -> int:
    s = settings()
    failures: list[str] = []

    async def check(name: str, coro):
        try:
            print(f"  {name:<22} {await coro}")
        except Exception as exc:  # noqa: BLE001 - report all, fail at the end
            print(f"  {name:<22} FAIL: {type(exc).__name__}: {exc}")
            failures.append(name)

    print("library-agent healthcheck\n")

    async def _pg():
        async with session_scope() as db:
            ver = (
                await db.execute(text("select extversion from pg_extension where extname='vector'"))
            ).scalar()
            tables = (
                await db.execute(
                    text(
                        "select count(*) from information_schema.tables where table_schema='public'"
                    )
                )
            ).scalar()
            if not ver:
                raise RuntimeError("pgvector extension not installed")
            return f"ok (pgvector {ver}, {tables} tables)"

    async def _roundtrip():
        vec = await embed_query("a test of the embedding round trip")
        async with session_scope() as db:
            got = (
                await db.execute(text("select (:v)::halfvec <=> (:v)::halfvec"), {"v": str(vec)})
            ).scalar()
            return f"ok (dim {len(vec)}, self-distance {got:.6f})"

    async def _redis():
        r = redis_client()
        try:
            ok, reason = await reading_may_proceed(r)
            return f"ok (reading_may_proceed={ok}, {reason or 'idle'})"
        finally:
            await r.aclose()

    async def _ollama():
        async with LLM() as c:
            loaded = await c.loaded_models()
            return f"ok (resident: {', '.join(loaded) or 'none'})"

    await check("postgres+pgvector", _pg())
    await check("embedding roundtrip", _roundtrip())
    await check("redis lease", _redis())
    await check("ollama", _ollama())

    print(f"\nmodels: reader={s.reader_model} chat={s.chat_model} embed={s.embed_model}")
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
