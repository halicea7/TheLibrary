"""`./library export` and `./library insert` -- cartridges from the terminal."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from sqlalchemy import select

from library_agent.db.models import Category
from library_agent.db.session import session_scope
from library_agent.library import cartridge as cart


async def _export(args: argparse.Namespace) -> None:
    async with session_scope() as db:
        cat_ids: list[uuid.UUID] = []
        for name in args.subject or []:
            row = (
                await db.execute(select(Category).where(Category.name.ilike(name)))
            ).scalar_one_or_none()
            if not row:
                raise SystemExit(f"no subject named {name!r}")
            cat_ids.append(row.id)
        ids = await cart.resolve_selection(
            db,
            category_ids=cat_ids,
            document_ids=[uuid.UUID(x) for x in args.document or []],
            cartridge_ids=[uuid.UUID(x) for x in args.cartridge or []],
        )
        if not ids:
            raise SystemExit("nothing selected; give --subject, --document or --cartridge")
        path = await cart.build_cartridge(
            db,
            document_ids=ids,
            level=args.level,
            name=args.name,
            colour=args.colour,
            made_by=args.made_by,
            out_dir=Path(args.out) if args.out else None,
        )
    m = cart.read_manifest(path)
    c = m["counts"]
    print(
        f"{path}\n  {m['name']} · {m['level']} · {c['documents']} documents, "
        f"{c['chunks']} passages, {c['artifacts']} readings, {c['vectors']} vectors"
    )


async def _insert(args: argparse.Namespace) -> None:
    from arq import create_pool
    from arq.connections import RedisSettings

    from library_agent.config import settings
    from library_agent.worker.tasks import schedule_library_rebuild

    async with session_scope() as db:
        res = await cart.import_cartridge(db, Path(args.zip))
    if res.noop:
        print(f"{res.name} v{res.version} is already on the rack")
        return
    print(
        f"inserted {res.name} v{res.version}: {res.documents_introduced} new, "
        f"{res.documents_joined} already here, {res.artifacts} readings "
        f"({res.vectors_loaded} vectors loaded, {res.vectors_embedded} re-embedded)"
    )
    redis = await create_pool(RedisSettings.from_dsn(settings().redis_url))
    try:
        await schedule_library_rebuild(redis)
    finally:
        await redis.aclose()


def main() -> None:
    ap = argparse.ArgumentParser(prog="library export|insert")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="make a cartridge")
    e.add_argument("--name", required=True)
    e.add_argument("--level", choices=[str(x) for x in cart.CartridgeLevel], default="readings")
    e.add_argument("--subject", action="append", help="a category name; repeatable")
    e.add_argument("--document", action="append", help="a document id; repeatable")
    e.add_argument("--cartridge", action="append", help="re-export a cartridge; repeatable")
    e.add_argument("--colour", default=None, help="#rrggbb")
    e.add_argument("--made-by", default=None)
    e.add_argument("--out", default=None, help="directory; default ~/.library-agent/cartridges")
    i = sub.add_parser("insert", help="put a cartridge on the rack")
    i.add_argument("zip")
    args = ap.parse_args()
    asyncio.run(_export(args) if args.cmd == "export" else _insert(args))


if __name__ == "__main__":
    main()
