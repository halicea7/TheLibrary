"""Bulk import from the filesystem.

    uv run python -m library_agent.ingest.bulk ~/papers ~/books --read

Walks directories, skips what is already shelved by content hash, and optionally queues
Tier 1 reading for everything it adds. For a first load this beats dragging hundreds of
files through the browser one at a time."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from library_agent.db.session import session_scope
from library_agent.ingest import dedup
from library_agent.ingest.extract import SUPPORTED, content_hash
from library_agent.ingest.tier0 import ingest
from library_agent.llm.ollama import Ollama


def find_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.suffix.lower() in SUPPORTED:
                out.append(root)
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith("."):
                out.append(p)
    return out


async def run(roots: list[Path], *, read: bool, quiet: bool) -> int:
    files = find_files(roots)
    if not files:
        print("nothing the library can read under", ", ".join(str(r) for r in roots))
        return 1
    print(f"{len(files)} files")

    added: list = []
    dupes = failed = 0
    started = time.time()
    async with Ollama() as client:
        for i, path in enumerate(files, 1):
            # Cheap pre-check so re-running over the same tree is nearly free.
            async with session_scope() as db:
                if await dedup.find_exact(db, content_hash(path)):
                    dupes += 1
                    if not quiet:
                        print(f"  [{i:>4}/{len(files)}] already shelved  {path.name}")
                    continue
            try:
                async with session_scope() as db:
                    r = await ingest(db, path, original_filename=path.name, client=client)
                added.append(r.document_id)
                if not quiet:
                    print(f"  [{i:>4}/{len(files)}] {r.chunks:4d} passages  {r.title[:60]}")
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the load
                failed += 1
                print(
                    f"  [{i:>4}/{len(files)}] FAILED  {path.name}: {str(exc)[:80]}", file=sys.stderr
                )

    print(
        f"\n{len(added)} shelved · {dupes} already there · {failed} unreadable · "
        f"{time.time() - started:.0f}s"
    )

    if read and added:
        from arq import create_pool
        from arq.connections import RedisSettings

        from library_agent.config import settings
        from library_agent.worker.tasks import enqueue_read

        redis = await create_pool(RedisSettings.from_dsn(settings().redis_url))
        try:
            for did in added:
                await enqueue_read(redis, did)
        finally:
            await redis.aclose()
        print(f"{len(added)} queued for reading (the worker must be running: ./library)")
    return 0 if not failed or added else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="import files and folders into the library")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--read", action="store_true", help="queue Tier 1 reading for new documents")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    missing = [p for p in a.paths if not p.exists()]
    if missing:
        raise SystemExit(f"not found: {', '.join(str(m) for m in missing)}")
    raise SystemExit(asyncio.run(run(a.paths, read=a.read, quiet=a.quiet)))


if __name__ == "__main__":
    main()
