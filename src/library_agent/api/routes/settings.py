"""Settings: what the library is running on, maintenance, and the incident log with the
model's troubleshooting. Configuration is environment-driven and shown read-only here,
each value with the variable that changes it."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from library_agent.config import settings
from library_agent.db.models import Document
from library_agent.db.purge import count_orphans, gc_orphan_vectors, gc_store
from library_agent.db.session import SessionDep
from library_agent.llm import providers, rerank
from library_agent.llm.liveness import gate, liveness
from library_agent.llm.openai_compat import OpenAICompat
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
            "model_answering": liveness.alive,
            "model_liveness": liveness.snapshot(),
            "generations": gate.snapshot(),
            "ollama_url": cfg.ollama_url,
            "resident_models": resident,
            "reranker": rerank.available(),
        },
        "models": [
            item(
                "Embeddings",
                f"{cfg.embed_model} · {cfg.embed_dim}d",
                "EMBED_MODEL",
                "changing it means re-embedding everything",
            ),
            item("Reranker", cfg.reranker_model, "RERANKER_MODEL", "runs locally, torch/MPS"),
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


# --- providers and the models each role runs on -------------------------------------

_catalogue: dict[str, tuple[float, list[str]]] = {}


async def _models_on(pid: str, prov: providers.Provider | None) -> list[str]:
    """What a backend offers, remembered for a minute; a backend that will not answer
    offers nothing rather than delaying the page."""
    now = time.monotonic()
    hit = _catalogue.get(pid)
    if hit and now - hit[0] < 60:
        return hit[1]
    names: list[str] = []
    try:
        if prov is None:
            async with httpx.AsyncClient(base_url=settings().ollama_url, timeout=3) as c:
                r = await c.get("/api/tags")
                names = sorted(m["name"] for m in r.json().get("models", []))
        else:
            async with OpenAICompat(prov) as c:
                names = await c.models()
    except Exception:  # noqa: BLE001
        names = []
    _catalogue[pid] = (now, names)
    return names


@router.get("/providers")
async def list_providers() -> dict:
    cfg = providers.load()
    catalogue = {"ollama": await _models_on("ollama", None)}
    for pid, prov in cfg.providers.items():
        catalogue[pid] = [f"{pid}:{m}" for m in await _models_on(pid, prov)]
    roles = []
    for rid, meta in providers.ROLES.items():
        model = providers.model_for(rid)
        roles.append(
            {
                "id": rid,
                "label": meta["label"],
                "note": meta["note"],
                "model": model,
                "default": providers.default_for(rid),
                "overridden": rid in cfg.models,
                "remote": providers.is_remote(model) if model else False,
            }
        )
    return {
        "providers": [p.public() for p in cfg.providers.values()],
        "roles": roles,
        "catalogue": catalogue,
        "file": str(providers.path()).replace(str(Path.home()), "~"),
    }


class ProviderIn(BaseModel):
    name: str = ""
    base_url: str
    # Omitted or null keeps the key already on file; "" clears it.
    api_key: str | None = None
    headers: dict[str, str] = {}


@router.put("/providers/{pid}")
async def put_provider(pid: str, body: ProviderIn) -> dict:
    if not providers.valid_id(pid):
        raise HTTPException(422, "an id is lowercase letters, digits and dashes, 32 at most")
    if pid == "ollama":
        raise HTTPException(422, "ollama is the default and is set by LIBRARY_OLLAMA_URL")
    base = body.base_url.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise HTTPException(422, "the base URL starts with http:// or https://")
    cfg = providers.load()
    old = cfg.providers.get(pid)
    key = body.api_key if body.api_key is not None else (old.api_key if old else "")
    cfg.providers[pid] = providers.Provider(
        id=pid, name=body.name.strip() or pid, base_url=base, api_key=key, headers=body.headers
    )
    providers.save(cfg)
    _catalogue.pop(pid, None)
    return cfg.providers[pid].public()


@router.delete("/providers/{pid}")
async def delete_provider(pid: str) -> dict:
    cfg = providers.load()
    if pid not in cfg.providers:
        raise HTTPException(404, "no such provider")
    del cfg.providers[pid]
    # Roles that pointed at it go back to their defaults.
    freed = [r for r, m in cfg.models.items() if m.startswith(pid + ":")]
    for r in freed:
        del cfg.models[r]
    providers.save(cfg)
    _catalogue.pop(pid, None)
    return {"deleted": pid, "roles_reset": freed}


@router.post("/providers/{pid}/test")
async def test_provider(pid: str, model: str | None = None) -> dict:
    prov = providers.load().providers.get(pid)
    if not prov:
        raise HTTPException(404, "no such provider")
    if model and model.startswith(pid + ":"):
        model = model[len(pid) + 1 :]
    t0 = time.monotonic()
    async with OpenAICompat(prov) as c:
        out = await c.probe(model)
    out["seconds"] = round(time.monotonic() - t0, 2)
    if out["models"]:
        _catalogue[pid] = (time.monotonic(), out["models"])
    return out


@router.put("/models")
async def put_models(body: dict[str, str | None]) -> dict:
    """Assign roles: {role: "model"} to override, {role: null} to return to the default.
    Reading is included on purpose and the page says what changing it costs."""
    cfg = providers.load()
    for role, model in body.items():
        if role not in providers.ROLES:
            raise HTTPException(422, f"no role called {role}")
        if model is None:
            cfg.models.pop(role, None)
            continue
        model = model.strip()
        prov, _ = providers.split(model)
        if ":" in model and prov is None and model.split(":", 1)[0] in cfg.providers:
            raise HTTPException(422, f"{model} names no model on that provider")
        cfg.models[role] = model
    providers.save(cfg)
    return {r: providers.model_for(r) for r in providers.ROLES}


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
