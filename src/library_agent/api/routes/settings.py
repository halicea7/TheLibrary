"""Settings: what the library is running on, maintenance, and the incident log with the
model's troubleshooting. Configuration is environment-driven and shown read-only here,
each value with the variable that changes it."""

from __future__ import annotations

import uuid

import httpx
from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from library_agent.config import settings
from library_agent.db.models import Document
from library_agent.db.purge import count_orphans, gc_orphan_vectors, gc_store
from library_agent.db.session import SessionDep
from library_agent.llm import rerank
from library_agent.ops import incidents

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
async def show(db: SessionDep) -> dict:
    cfg = settings()
    # Services, checked live and cheaply.
    ollama_ok, resident = False, []
    try:
        async with httpx.AsyncClient(base_url=cfg.ollama_url, timeout=3) as c:
            ollama_ok = (await c.get("/api/tags")).status_code == 200
            ps = await c.get("/api/ps")
            resident = (
                [m["name"] for m in ps.json().get("models", [])] if ps.status_code == 200 else []
            )
    except Exception:  # noqa: BLE001
        ollama_ok = False
    redis_ok = False
    try:
        from redis.asyncio import Redis

        r = Redis.from_url(cfg.redis_url)
        try:
            redis_ok = bool(await r.ping())
        finally:
            await r.aclose()
    except Exception:  # noqa: BLE001
        redis_ok = False

    def item(name: str, value, env: str, note: str = "") -> dict:
        return {"name": name, "value": value, "env": f"LIBRARY_{env}", "note": note}

    return {
        "services": {
            "postgres": True,  # we answered this request through it
            "redis": redis_ok,
            "ollama": ollama_ok,
            "ollama_url": cfg.ollama_url,
            "resident_models": resident,
            "reranker": rerank.available(),
        },
        "models": [
            item(
                "Reading",
                cfg.reader_model,
                "READER_MODEL",
                "pinned; artifacts record which model wrote them",
            ),
            item(
                "Chat (general)",
                cfg.chat_model_options.get("general", cfg.chat_model),
                "CHAT_MODEL_OPTIONS",
                "switchable per conversation",
            ),
            item(
                "Chat (technical)",
                cfg.chat_model_options.get("technical"),
                "CHAT_MODEL_OPTIONS",
                "",
            ),
            item(
                "Embeddings",
                f"{cfg.embed_model} · {cfg.embed_dim}d",
                "EMBED_MODEL",
                "changing it means re-embedding everything",
            ),
            item("Reranker", cfg.reranker_model, "RERANKER_MODEL", "runs locally, torch/MPS"),
            item(
                "Troubleshooting",
                cfg.troubleshoot_model
                or cfg.chat_model_options.get("technical")
                or cfg.reader_model,
                "TROUBLESHOOT_MODEL",
                "reads the incident against the docs",
            ),
            item("Keep alive", cfg.keep_alive, "KEEP_ALIVE", "-1 pins models resident"),
        ],
        "retrieval": [
            item("Passages per answer", cfg.chat_passages, "CHAT_PASSAGES"),
            item("Candidate pool", cfg.candidate_pool, "CANDIDATE_POOL"),
            item("Lexical weight", cfg.weight_lexical, "WEIGHT_LEXICAL", "dense is 1.0"),
            item("Chunk target", f"{cfg.chunk_target_tokens} tokens", "CHUNK_TARGET_TOKENS"),
        ],
        "library": [
            item(
                "Threads rebuild after",
                f"{cfg.library_rebuild_delay_seconds}s quiet",
                "LIBRARY_REBUILD_DELAY_SECONDS",
            ),
            item(
                "Incident window",
                f"{cfg.incident_dedupe_seconds}s",
                "INCIDENT_DEDUPE_SECONDS",
                "same error inside it counts up",
            ),
            item("Repository", cfg.repo_url, "REPO_URL", "where 'open an issue' goes"),
        ],
        "storage": {
            "store": str(cfg.storage_dir).replace(str(cfg.storage_dir.home()), "~"),
            "database": cfg.database_url.split("@")[-1],
            "orphans": await count_orphans(db),
        },
        "incidents_open": await incidents.open_count(db),
    }


@router.post("/gc")
async def gc(db: SessionDep) -> dict:
    """Sweep orphaned vectors and stray files. Safe any time."""
    vectors = await gc_orphan_vectors(db)
    hashes = set((await db.execute(select(Document.content_hash))).scalars())
    files = gc_store(hashes)
    await db.commit()
    return {"vectors_removed": vectors, "files_removed": files}


@router.get("/incidents")
async def list_incidents(db: SessionDep, resolved: bool = False) -> list[dict]:
    rows = await incidents.list_incidents(db, include_resolved=resolved)
    return [
        {
            "id": str(i.id),
            "source": i.source,
            "kind": i.kind,
            "message": i.message,
            "detail": i.detail,
            "context": i.context,
            "count": i.count,
            "first_at": i.first_at.isoformat() if i.first_at else None,
            "last_at": i.last_at.isoformat() if i.last_at else None,
            "resolved": i.resolved,
            "advice": i.advice,
            "advice_at": i.advice_at.isoformat() if i.advice_at else None,
        }
        for i in rows
    ]


@router.post("/incidents/{incident_id}/advise")
async def advise(incident_id: uuid.UUID, db: SessionDep) -> dict:
    """Ask the model to troubleshoot this incident against the library's own docs. It
    returns steps for the person to run; nothing is executed here."""
    try:
        advice = await incidents.advise(db, incident_id)
    except KeyError as exc:
        raise HTTPException(404, "no such incident") from exc
    await db.commit()
    return advice


@router.post("/incidents/{incident_id}/resolve")
async def resolve(incident_id: uuid.UUID, db: SessionDep, resolved: bool = True) -> dict:
    await incidents.resolve(db, incident_id, resolved=resolved)
    await db.commit()
    return {"resolved": resolved}


@router.post("/incidents/test")
async def raise_test_incident() -> dict:
    """Record a harmless incident so the panel can be tried."""
    try:
        raise RuntimeError("test incident: nothing is wrong, this was raised on request")
    except RuntimeError as exc:
        await incidents.record_exception(
            exc, source="api", context={"route": "/api/settings/incidents/test"}
        )
    return {"ok": True}
