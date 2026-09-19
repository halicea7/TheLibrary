"""Shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
async def db():
    """A session whose work is thrown away."""
    try:
        from library_agent.db.session import engine
    except Exception:  # noqa: BLE001
        pytest.skip("no database")
    from sqlalchemy.ext.asyncio import AsyncSession

    async with engine().connect() as conn:
        tx = await conn.begin()
        s = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            yield s
        finally:
            await s.close()
            await tx.rollback()
