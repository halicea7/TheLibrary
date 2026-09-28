"""The connectors that ship with the library, as data like any other: manifests in
`modules/manifests/`, parsed by the same checks as one you write. SentinelOne is the
first: five read-only operations against the v2.1 REST API -- whether an indicator has
been seen, who is exposed to a CVE, whether an application is in the fleet and where, and
the state of the agents. Every path is a GET; the shapes are verified against the
tenant's own mock and api_docs 2.1."""

from __future__ import annotations

import json
from pathlib import Path

from library_agent.modules.manifest import Module, from_dict

MANIFESTS = Path(__file__).parent / "manifests"


def _load() -> dict[str, Module]:
    out: dict[str, Module] = {}
    for p in sorted(MANIFESTS.glob("*.json")):
        m = from_dict(json.loads(p.read_text()), source=p.name, builtin=True)
        out[m.id] = m
    return out


BUILTIN: dict[str, Module] = _load()
SENTINELONE = BUILTIN["sentinelone"]
