"""Where a module's operator settings live: its base URL, its token, and whether it is
seated. The operations are the manifest's (code, for a built-in); only these three are the
operator's. The file sits beside the provider keys, 0600, and the token is never returned
to the browser or shown to the model -- only whether one is set, and its last four."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from library_agent.config import settings
from library_agent.modules.builtin import BUILTIN
from library_agent.modules.manifest import Module


@dataclass
class ModuleConfig:
    id: str
    base_url: str = ""
    token: str = ""
    seated: bool = False  # consulted during a question, the way a seated cartridge scopes
    extra_headers: dict[str, str] = field(default_factory=dict)

    def public(self) -> dict:
        return {
            "id": self.id,
            "base_url": self.base_url,
            "seated": self.seated,
            "has_token": bool(self.token),
            "token_tail": self.token[-4:] if len(self.token) >= 8 else "",
            "configured": bool(self.base_url and self.token),
        }


def path() -> Path:
    override = os.environ.get("LIBRARY_MODULES_FILE")
    return Path(override) if override else settings().storage_dir.parent / "modules.json"


_cache: tuple[Path, float, dict[str, ModuleConfig]] | None = None


def load() -> dict[str, ModuleConfig]:
    """Config by module id, read once per change. A missing file is no configured modules."""
    global _cache
    p = path()
    try:
        mtime = p.stat().st_mtime
    except OSError:
        _cache = None
        return {}
    if _cache and _cache[0] == p and _cache[1] == mtime:
        return _cache[2]
    out: dict[str, ModuleConfig] = {}
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    for mid, v in (raw.get("modules") or {}).items():
        if mid in BUILTIN and isinstance(v, dict):
            out[mid] = ModuleConfig(
                id=mid,
                base_url=str(v.get("base_url") or "").rstrip("/"),
                token=str(v.get("token") or ""),
                seated=bool(v.get("seated")),
                extra_headers={str(k): str(x) for k, x in (v.get("extra_headers") or {}).items()},
            )
    _cache = (p, mtime, out)
    return out


def save(configs: dict[str, ModuleConfig]) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "modules": {
            c.id: {
                "base_url": c.base_url,
                "token": c.token,
                "seated": c.seated,
                "extra_headers": c.extra_headers,
            }
            for c in configs.values()
        }
    }
    tmp = p.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)
    now = time.time()
    os.utime(p, (now, now))


def config_for(mid: str) -> ModuleConfig:
    return load().get(mid, ModuleConfig(id=mid))


def seated_modules() -> list[tuple[Module, ModuleConfig]]:
    """The modules the librarian should consult: seated, with a base URL and a token."""
    cfgs = load()
    out = []
    for mid, mod in BUILTIN.items():
        c = cfgs.get(mid)
        if c and c.seated and c.base_url and c.token:
            out.append((mod, c))
    return out
