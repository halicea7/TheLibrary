"""Every connector the library knows: the built-ins, and the manifests in
`~/.library-agent/connectors/` (or $LIBRARY_CONNECTORS_DIR). A manifest that does not
pass its checks is not loaded; the reason is kept so Settings and the bay can say why.
Reloaded when a file in the folder changes."""

from __future__ import annotations

import json
import os
from pathlib import Path

from library_agent.config import settings
from library_agent.modules.builtin import BUILTIN
from library_agent.modules.manifest import ManifestError, Module, from_dict

_cache: tuple[tuple, dict[str, Module], dict[str, str]] | None = None


def folder() -> Path:
    override = os.environ.get("LIBRARY_CONNECTORS_DIR")
    d = Path(override) if override else settings().storage_dir.parent / "connectors"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stamp(d: Path) -> tuple:
    return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in d.glob("*.json")))


def _load() -> tuple[dict[str, Module], dict[str, str]]:
    global _cache
    d = folder()
    stamp = (str(d), _stamp(d))
    if _cache and _cache[0] == stamp:
        return _cache[1], _cache[2]
    mods: dict[str, Module] = dict(BUILTIN)
    errors: dict[str, str] = {}
    for p in sorted(d.glob("*.json")):
        try:
            m = from_dict(json.loads(p.read_text()), source=p.name)
        except (ManifestError, ValueError, OSError) as exc:
            errors[p.name] = str(exc)
            continue
        if m.id in mods:
            errors[p.name] = f"id {m.id!r} is already taken"
            continue
        mods[m.id] = m
    _cache = (stamp, mods, errors)
    return mods, errors


def all_modules() -> dict[str, Module]:
    return _load()[0]


def get(mid: str) -> Module | None:
    return all_modules().get(mid)


def errors() -> dict[str, str]:
    """Manifest files that did not load, and why."""
    return _load()[1]


def path_for(mid: str) -> Path:
    return folder() / f"{mid}.json"


def forget() -> None:
    global _cache
    _cache = None
