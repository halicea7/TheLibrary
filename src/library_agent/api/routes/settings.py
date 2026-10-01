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
    # How far it is trusted: the highest classification level it may see ("" = the remote
    # ceiling), and whether it runs on infrastructure you control.
    ceiling: str = ""
    internal: bool = False


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
    if body.ceiling:
        from library_agent import classification

        if body.ceiling not in {lv.id for lv in classification.load().levels}:
            raise HTTPException(422, "the ceiling is a level on the classification scale")
    cfg.providers[pid] = providers.Provider(
        id=pid,
        name=body.name.strip() or pid,
        base_url=base,
        api_key=key,
        headers=body.headers,
        ceiling=body.ceiling,
        internal=body.internal,
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


class About(BaseModel):
    about: str


@router.get("/about")
async def get_about() -> dict:
    return {"about": providers.load().about, "max": providers.ABOUT_MAX}


@router.put("/about")
async def put_about(body: About) -> dict:
    """Who the library is for. Read into every answer, so it is kept short."""
    if len(body.about) > providers.ABOUT_MAX:
        raise HTTPException(422, f"at most {providers.ABOUT_MAX} characters")
    cfg = providers.load()
    cfg.about = body.about.strip()
    providers.save(cfg)
    return {"about": cfg.about, "max": providers.ABOUT_MAX}


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


@router.post("/incidents/resolve-all")
async def resolve_all(db: SessionDep) -> dict:
    n = await incidents.resolve_all(db)
    await db.commit()
    return {"resolved": n}


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


# --- modules: lines to live sources -------------------------------------------------


class ModulePut(BaseModel):
    base_url: str | None = None
    token: str | None = None  # kept if omitted, so the UI never has to re-send it
    username: str | None = None  # basic auth's user, or an OAuth client id
    seated: bool | None = None
    # The bay: 0-3 puts the token in that socket (seating it), -1 takes it out.
    socket: int | None = None
    design: dict | None = None
    # A level id on the classification scale; "" returns it to the default.
    classification: str | None = None


def _module_public(mod, cfg) -> dict:
    return {
        "id": mod.id,
        "name": mod.name,
        "kind": mod.kind,
        "colour": mod.colour,
        "description": mod.description,
        "local_only": mod.local_only,
        "auth_scheme": mod.auth_scheme,
        "operations": [
            {
                "id": o.id,
                "summary": o.summary,
                "ask_when": o.ask_when,
                "params": list((o.params.get("properties") or {}).keys()),
            }
            for o in mod.operations
        ],
        **cfg.public(),
        "configured": mod_store_configured(mod, cfg),
        "auth": mod.auth_spec().type,
        "builtin": mod.builtin,
        "suggested_base_url": mod.base_url,
    }


def mod_store_configured(mod, cfg) -> bool:
    from library_agent.modules.store import is_configured

    return is_configured(mod, cfg)


@router.get("/modules")
async def list_modules() -> dict:
    from library_agent.modules import store as mod_store
    from library_agent.modules.registry import all_modules

    cfgs = mod_store.load()
    mods = [
        _module_public(mod, cfgs.get(mid) or mod_store.ModuleConfig(id=mid))
        for mid, mod in all_modules().items()
    ]
    return {"modules": mods, "sockets": mod_store.SOCKETS}


@router.put("/modules/{mid}")
async def put_module(mid: str, body: ModulePut) -> dict:
    from library_agent.modules import store as mod_store
    from library_agent.modules.registry import all_modules

    BUILTIN = all_modules()
    if mid not in BUILTIN:
        raise HTTPException(404, f"no module {mid!r}")
    cfgs = mod_store.load()
    cfg = cfgs.get(mid) or mod_store.ModuleConfig(id=mid)
    if body.base_url is not None:
        cfg.base_url = body.base_url.strip().rstrip("/")
    if body.token is not None and body.token != "":
        cfg.token = body.token.strip()
    if body.username is not None:
        cfg.username = body.username.strip()[:200]
    taken = {c.socket: c.id for c in cfgs.values() if c.socket is not None and c.id != mid}
    want = body.socket
    if want is None and body.seated is not None:
        # Seated from a checkbox: the first free socket, or out.
        want = (
            (
                cfg.socket
                if cfg.socket is not None
                else next((i for i in range(mod_store.SOCKETS) if i not in taken), None)
            )
            if body.seated
            else -1
        )
        if want is None:
            raise HTTPException(
                409, f"all {mod_store.SOCKETS} sockets are taken; take a token out first"
            )
    if want is not None:
        if want == -1:
            cfg.socket, cfg.seated = None, False
        elif 0 <= want < mod_store.SOCKETS:
            if want in taken:
                raise HTTPException(409, f"socket {want + 1} holds {taken[want]}")
            cfg.socket, cfg.seated = want, True
        else:
            raise HTTPException(422, f"socket is 0-{mod_store.SOCKETS - 1}, or -1 to take it out")
    if body.design is not None:
        from library_agent.modules.design import DesignError, clean

        try:
            cfg.design = clean(body.design)
        except DesignError as exc:
            raise HTTPException(422, str(exc)) from exc
    if body.classification is not None:
        from library_agent import classification as cls

        if body.classification and body.classification not in {lv.id for lv in cls.load().levels}:
            raise HTTPException(422, "not a level on the classification scale")
        cfg.classification = body.classification
    cfgs[mid] = cfg
    mod_store.save(cfgs)
    return _module_public(BUILTIN[mid], cfg)


@router.post("/modules/{mid}/test")
async def test_module(mid: str) -> dict:
    """Reach the module with its configured base URL and token, so a wrong URL or token
    shows at once. It runs the lightest operation it has -- one that needs no input, if
    there is one (listing agents, say) -- rather than the first: on a large SentinelOne
    tenant the first, a CVE search over millions of risk rows, takes 20-50 seconds, and
    a test that sits on "reaching" for a minute looks broken when it isn't."""
    from library_agent.modules import store as mod_store
    from library_agent.modules.execute import ModuleError, call
    from library_agent.modules.registry import all_modules

    BUILTIN = all_modules()
    if mid not in BUILTIN:
        raise HTTPException(404, f"no module {mid!r}")
    mod = BUILTIN[mid]
    cfg = mod_store.config_for(mid)
    if not (cfg.base_url and cfg.token):
        return {"ok": False, "error": "set a base URL and a token first"}
    op = next(
        (o for o in mod.operations if not (o.params.get("required") or [])),
        mod.operations[0],
    )
    probe = {
        k: "CVE-0000-0000" if "cve" in k else "___probe___"
        for k in (op.params.get("required") or [])
    }
    try:
        res = await call(mod, cfg, op, probe)
        return {"ok": True, "reached": True, "count": res["count"], "operation": op.id}
    except ModuleError as exc:
        return {"ok": False, "error": str(exc)}
