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


def looks_like_prose(path: Path, *, sample_bytes: int = 6000) -> bool:
    """Wordlists, payload dumps and fuzz lists are text files but not reading material:
    hundreds of short lines with almost no sentences. Skip them rather than shelve them --
    a single path-traversal list produced 1,101 'passages' before this check existed."""
    try:
        if path.stat().st_size < 200:
            return False  # EICAR is 68 bytes; no reading material is this small
    except OSError:
        return False
    if path.suffix.lower() != ".txt":
        return True
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:sample_bytes]
    except OSError:
        return False
    lines = [ln for ln in head.splitlines() if ln.strip()]
    if len(lines) < 12:
        return True
    avg = sum(len(ln) for ln in lines) / len(lines)
    with_space = sum(1 for ln in lines if " " in ln.strip()) / len(lines)
    body = "".join(lines)
    symbols = sum(1 for ch in body if not ch.isalnum() and not ch.isspace()) / max(1, len(body))
    # Prose lines are long, contain spaces, and are mostly letters. Payloads and
    # polyglots can be long and spaced too -- but they are dense with < > / % ' " ( ).
    return (avg >= 40 or with_space >= 0.6) and symbols < 0.12


def find_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.suffix.lower() in SUPPORTED:
                out.append(root)
            continue
        for p in sorted(root.rglob("*")):
            if (
                p.is_file()
                and p.suffix.lower() in SUPPORTED
                and not p.name.startswith(".")
                and looks_like_prose(p)
            ):
                out.append(p)
    return out


async def _cartridge_for(name: str, colour: str | None):
    """A cartridge made on this machine from a folder: it stands on the rack like an
    inserted one, and its volumes are introduced by it, so ejecting it takes them along."""
    import uuid
    from datetime import UTC, datetime

    from library_agent.config import settings
    from library_agent.db.models import Cartridge
    from library_agent.library.cartridge import PALETTE, slugify

    slug = slugify(name)
    async with session_scope() as db:
        from sqlalchemy import select

        row = (await db.execute(select(Cartridge).where(Cartridge.slug == slug))).scalars().first()
        if row:
            return row.id
        n = len(list((await db.execute(select(Cartridge.id))).scalars()))
        row = Cartridge(
            id=uuid.uuid4(),
            name=name.strip(),
            slug=slug,
            version=1,
            colour=colour or PALETTE[n % len(PALETTE)],
            made_by="import",
            made_at=datetime.now(UTC),
            level="full",
            embed_model=settings().embed_model,
            reader_model=settings().reader_model,
            prompt_versions=dict(settings().prompt_versions),
            manifest={"origin": "import"},
            content_hash="local",
        )
        db.add(row)
        await db.flush()
        return row.id


async def run(
    roots: list[Path],
    *,
    read: bool,
    quiet: bool,
    cartridge: str | None = None,
    colour: str | None = None,
) -> int:
    files = find_files(roots)
    if not files:
        print("nothing the library can read under", ", ".join(str(r) for r in roots))
        return 1
    print(f"{len(files)} files")

    added: list = []
    dupes = failed = 0
    started = time.time()
    cart_id = await _cartridge_for(cartridge, colour) if cartridge else None

    async def member(db, document_id, introduced: bool) -> None:
        if cart_id:
            from library_agent.db.models import CartridgeDocument

            await db.merge(
                CartridgeDocument(
                    cartridge_id=cart_id, document_id=document_id, introduced=introduced
                )
            )

    async with Ollama() as client:
        for i, path in enumerate(files, 1):
            # Cheap pre-check so re-running over the same tree is nearly free.
            async with session_scope() as db:
                if existing := await dedup.find_exact(db, content_hash(path)):
                    dupes += 1
                    await member(db, existing.id, False)
                    if not quiet:
                        print(f"  [{i:>4}/{len(files)}] already shelved  {path.name}")
                    continue
            try:
                async with session_scope() as db:
                    r = await ingest(db, path, original_filename=path.name, client=client)
                    if r.status == "ready":
                        await member(db, r.document_id, True)
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
        f"{time.time() - started:.0f}s" + (f" · on the rack as {cartridge!r}" if cartridge else "")
    )
    if cart_id:
        from sqlalchemy import func, select

        from library_agent.db.models import Cartridge, CartridgeDocument

        async with session_scope() as db:
            n = (
                await db.execute(
                    select(func.count()).where(CartridgeDocument.cartridge_id == cart_id)
                )
            ).scalar_one()
            row = await db.get(Cartridge, cart_id)
            if row:
                row.document_count = n

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
    ap.add_argument(
        "--cartridge", help="put the imported volumes on the rack as a cartridge of this name"
    )
    ap.add_argument("--colour", help="#rrggbb for that cartridge")
    a = ap.parse_args()
    missing = [p for p in a.paths if not p.exists()]
    if missing:
        raise SystemExit(f"not found: {', '.join(str(m) for m in missing)}")
    raise SystemExit(
        asyncio.run(
            run(a.paths, read=a.read, quiet=a.quiet, cartridge=a.cartridge, colour=a.colour)
        )
    )


if __name__ == "__main__":
    main()
