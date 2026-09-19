"""FastAPI application."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from sqlalchemy import func, select

from library_agent.api.routes import cartridges, chat, documents, library, reading, search
from library_agent.api.schemas import HealthOut
from library_agent.config import settings
from library_agent.db.models import Chunk, Document, Embedding
from library_agent.db.purge import count_orphans
from library_agent.db.session import SessionDep, session_scope
from library_agent.llm.ollama import Ollama

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings().storage_dir.mkdir(parents=True, exist_ok=True)
    # Load the cross-encoder off the request path. It takes ~45s from cold and would
    # otherwise be paid by whoever runs the first reranked query.
    task = asyncio.create_task(asyncio.to_thread(_warm_reranker))
    yield
    task.cancel()


def _warm_reranker() -> None:
    from library_agent.llm import rerank

    if rerank.available():
        try:
            rerank.warm()
        except Exception:
            log.warning("reranker warmup failed; first reranked query will be slow", exc_info=True)


app = FastAPI(title="Library Agent", version="0.1.0", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(reading.router)
app.include_router(chat.router)
app.include_router(library.router)
app.include_router(cartridges.router)


@app.get("/api/health", response_model=HealthOut)
async def health(db: SessionDep) -> HealthOut:
    docs = (await db.execute(select(func.count()).select_from(Document))).scalar() or 0
    chunks = (await db.execute(select(func.count()).select_from(Chunk))).scalar() or 0
    embs = (await db.execute(select(func.count()).select_from(Embedding))).scalar() or 0
    orphans = await count_orphans(db)
    hashes = set((await db.execute(select(Document.content_hash))).scalars())
    store = settings().storage_dir
    stray = (
        sum(1 for f in store.rglob("*") if f.is_file() and f.stem not in hashes)
        if store.exists()
        else 0
    )
    orphans["stray_files"] = stray
    # Over an SSH tunnel this is the failure you actually hit: the tunnel drops and
    # every model call fails. Say so plainly rather than showing an empty model list.
    reachable = True
    try:
        async with Ollama() as c:
            resident = await c.loaded_models()
    except Exception:  # noqa: BLE001 - health must report, not raise
        resident, reachable = [], False
    cfg = settings()
    return HealthOut(
        ok=reachable and all(v == 0 for v in orphans.values()),
        documents=docs,
        chunks=chunks,
        embeddings=embs,
        orphan_vectors=orphans,
        models_resident=resident,
        embed_model=cfg.embed_model,
        chat_model=cfg.chat_model,
    )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


async def _warmup() -> None:
    async with session_scope() as db:
        await db.execute(select(func.count()).select_from(Document))
