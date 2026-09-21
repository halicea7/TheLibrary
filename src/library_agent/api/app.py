"""FastAPI application."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from library_agent.api.auth import BearerOrLoopback
from library_agent.api.routes import (
    cartridges,
    chat,
    compose,
    documents,
    library,
    reading,
    search,
    v1,
)
from library_agent.api.routes import (
    settings as settings_routes,
)
from library_agent.api.schemas import HealthOut
from library_agent.config import settings
from library_agent.db.models import Chunk, Document, Embedding
from library_agent.db.purge import count_orphans
from library_agent.db.session import SessionDep, session_scope
from library_agent.llm.liveness import gate, liveness
from library_agent.llm.ollama import Ollama
from library_agent.ops import incidents

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings().storage_dir.mkdir(parents=True, exist_ok=True)
    incidents.install_handler("api")
    liveness.start()
    # Load the cross-encoder off the request path. It takes ~45s from cold and would
    # otherwise be paid by whoever runs the first reranked query.
    task = asyncio.create_task(asyncio.to_thread(_warm_reranker))
    yield
    liveness.stop()
    task.cancel()


def _warm_reranker() -> None:
    from library_agent.llm import rerank

    if rerank.available():
        try:
            rerank.warm()
        except Exception:
            log.warning("reranker warmup failed; first reranked query will be slow", exc_info=True)


app = FastAPI(title="Library Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(BearerOrLoopback)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Anything that escapes a route becomes an incident the Settings tab can explain.
    HTTPExceptions (4xx the routes raise on purpose) never reach here."""
    await incidents.record_exception(
        exc, source="api", context={"method": request.method, "path": request.url.path}
    )
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"[:500]})


app.include_router(documents.router)
app.include_router(search.router)
app.include_router(reading.router)
app.include_router(chat.router)
app.include_router(library.router)
app.include_router(cartridges.router)
app.include_router(settings_routes.router)
app.include_router(v1.router)
app.include_router(compose.router)


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
        model_answering=liveness.alive,
        model_liveness=liveness.snapshot(),
        generations=gate.snapshot(),
        ok=reachable and all(v == 0 for v in orphans.values()),
        documents=docs,
        chunks=chunks,
        embeddings=embs,
        orphan_vectors=orphans,
        models_resident=resident,
        embed_model=cfg.embed_model,
        chat_model=cfg.chat_model,
    )


app.mount("/vendor", StaticFiles(directory=WEB_DIR / "vendor"), name="vendor")


@app.get("/cartridge3d.js", include_in_schema=False)
async def cartridge3d() -> FileResponse:
    return FileResponse(WEB_DIR / "cartridge3d.js", media_type="text/javascript")


@app.get("/nebula-gl.js", include_in_schema=False)
async def nebula_gl() -> FileResponse:
    return FileResponse(WEB_DIR / "nebula-gl.js", media_type="text/javascript")


@app.get("/demo.js", include_in_schema=False)
async def demo_js() -> FileResponse:
    """The stand-in server for the hosted demo; `?demo` on a local instance shows it."""
    return FileResponse(WEB_DIR / "demo.js", media_type="text/javascript")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


async def _warmup() -> None:
    async with session_scope() as db:
        await db.execute(select(func.count()).select_from(Document))
