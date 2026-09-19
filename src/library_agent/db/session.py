"""Async engine/session factory."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from library_agent.config import settings

_engine = create_async_engine(settings().database_url, pool_size=5, max_overflow=10)
SessionLocal = async_sessionmaker(_engine, expire_on_commit=False)


def engine():
    return _engine


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as s:
        try:
            yield s
            await s.commit()
        except Exception:
            await s.rollback()
            raise


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency."""
    async with SessionLocal() as s:
        yield s


# Annotated form keeps Depends() out of argument defaults, which is both the current
# FastAPI idiom and what flake8-bugbear's B008 wants.
SessionDep = Annotated[AsyncSession, Depends(get_session)]
