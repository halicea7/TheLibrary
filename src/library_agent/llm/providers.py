"""Other places a model can live.

Ollama is the default and needs nothing here. A provider is anything that speaks the
OpenAI chat-completions protocol -- OpenAI itself, Anthropic's compatible endpoint,
OpenRouter, Groq, Mistral, DeepSeek, LM Studio, vLLM, llama.cpp's server, Ollama's own
/v1 -- named once in the settings and then addressed by prefix: a model called
``openrouter:anthropic/claude-sonnet-4.5`` goes to the provider whose id is
``openrouter``; a model with no configured prefix goes to Ollama as it always did.

Roles are what the library does with a model. Each has a default from the environment
and may be pointed at any model, on Ollama or on a provider, from the settings. The
assignments and the providers live in one small file beside the document store, so the
API and the worker both see a change without a restart. Embeddings are not a role:
every vector in the library is bge-m3 at 1024 dimensions, and moving them is a
re-embedding of everything, not a setting.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from library_agent.config import settings

# What each role is for, in the words the settings page uses.
ROLES: dict[str, dict[str, str]] = {
    "chat_general": {
        "label": "Chat (general)",
        "note": "answers questions; switchable per conversation",
    },
    "chat_technical": {
        "label": "Chat (technical)",
        "note": "the coder model for technical questions",
    },
    "reading": {
        "label": "Reading",
        "note": "Tier 1 and 2; artifacts record their model, so a different one re-reads the whole shelf on the next backfill",
    },
    "threads": {"label": "Threads", "note": "cluster summaries, conflict judging, shelving"},
    "vision": {"label": "Vision", "note": "reads a PDF's figures; empty turns it off"},
    "troubleshoot": {"label": "Troubleshooting", "note": "reads an incident against the docs"},
}

_ID = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


@dataclass
class Provider:
    id: str
    name: str
    base_url: str
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        """For the UI: the key stays here, only its tail is shown."""
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "has_key": bool(self.api_key),
            "key_tail": self.api_key[-4:] if len(self.api_key) >= 8 else "",
            "headers": self.headers,
        }


@dataclass
class Config:
    providers: dict[str, Provider] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)  # role -> model, when overridden


def path() -> Path:
    override = os.environ.get("LIBRARY_PROVIDERS_FILE")
    return Path(override) if override else settings().storage_dir.parent / "providers.json"


_cache: tuple[Path, float, Config] | None = None


def load() -> Config:
    """Read the file, once per change. A missing or unreadable file is an empty config:
    the library runs on Ollama and the environment."""
    global _cache
    p = path()
    try:
        mtime = p.stat().st_mtime
    except OSError:
        _cache = None
        return Config()
    if _cache and _cache[0] == p and _cache[1] == mtime:
        return _cache[2]
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return Config()
    cfg = Config()
    for pid, v in (raw.get("providers") or {}).items():
        if _ID.match(pid) and v.get("base_url"):
            cfg.providers[pid] = Provider(
                id=pid,
                name=v.get("name") or pid,
                base_url=str(v["base_url"]).rstrip("/"),
                api_key=v.get("api_key") or "",
                headers={str(k): str(x) for k, x in (v.get("headers") or {}).items()},
            )
    cfg.models = {
        r: m for r, m in (raw.get("models") or {}).items() if r in ROLES and m is not None
    }
    _cache = (p, mtime, cfg)
    return cfg


def save(cfg: Config) -> None:
    """Written whole, to a private file, atomically: the key is in it."""
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "providers": {
            pid: {
                "name": v.name,
                "base_url": v.base_url,
                "api_key": v.api_key,
                "headers": v.headers,
            }
            for pid, v in cfg.providers.items()
        },
        "models": cfg.models,
    }
    tmp = p.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)
    # Make sure the next load sees a new mtime even on a coarse filesystem clock.
    now = time.time()
    os.utime(p, (now, now))


def valid_id(pid: str) -> bool:
    return bool(_ID.match(pid))


def split(model: str) -> tuple[Provider | None, str]:
    """``openrouter:foo/bar`` -> (the provider, "foo/bar") when ``openrouter`` is
    configured; anything else is an Ollama model, colons and all."""
    if ":" in model:
        head, tail = model.split(":", 1)
        prov = load().providers.get(head)
        if prov and tail:
            return prov, tail
    return None, model


def is_remote(model: str) -> bool:
    return split(model)[0] is not None


def default_for(role: str) -> str:
    """What the environment says, before any override."""
    cfg = settings()
    if role == "chat_general":
        return cfg.chat_model_options.get("general", cfg.chat_model)
    if role == "chat_technical":
        return cfg.chat_model_options.get("technical", "")
    if role == "reading":
        return cfg.reader_model
    if role == "threads":
        return cfg.reader_model
    if role == "vision":
        return cfg.vision_model
    if role == "troubleshoot":
        return cfg.troubleshoot_model or cfg.chat_model_options.get("technical") or cfg.reader_model
    raise KeyError(role)


def model_for(role: str) -> str:
    """The model a role runs on right now. An override of "" means none (vision off)."""
    m = load().models.get(role)
    return m if m is not None else default_for(role)


def chat_options() -> dict[str, str]:
    """The per-conversation choices, as the chat routes have always shaped them."""
    out = {"general": model_for("chat_general")}
    tech = model_for("chat_technical")
    if tech:
        out["technical"] = tech
    return out


def ollama_probe_model() -> str:
    """A model that lives on Ollama, for the liveness probe: the chat model if it is
    there, else the first role that is, else the reader."""
    for role in ("chat_general", "chat_technical", "reading", "threads"):
        m = model_for(role)
        if m and not is_remote(m):
            return m
    return settings().reader_model
