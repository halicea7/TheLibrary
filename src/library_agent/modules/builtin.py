"""The modules that ship with the library. The SentinelOne module is the first: five
read-only operations against the v2.1 REST API, the ones actually asked in a week —
whether an indicator has been seen, who is exposed to a CVE, whether an application is in
the fleet and where, and the state of the agents. Every path is a GET; the shapes are
verified against the tenant's own mock and api_docs 2.1."""

from __future__ import annotations

from library_agent.modules.manifest import Module, Operation, Render

_CVE = {
    "type": "object",
    "properties": {
        "cve": {"type": "string", "description": "a CVE id, e.g. CVE-2024-3094", "maxLength": 40}
    },
    "required": ["cve"],
}
_IOC = {
    "type": "object",
    "properties": {
        "indicator": {
            "type": "string",
            "description": "a hash, IP, domain or URL to look up",
            "maxLength": 400,
        }
    },
    "required": ["indicator"],
}
_APP = {
    "type": "object",
    "properties": {
        "application": {
            "type": "string",
            "description": "an application name, e.g. OpenSSL",
            "maxLength": 120,
        }
    },
    "required": ["application"],
}

SENTINELONE = Module(
    id="sentinelone",
    name="SentinelOne",
    kind="endpoint security",
    colour="#e0564b",  # its own live pigment, distinct from rubric
    auth_scheme="apitoken",
    auth_header="Authorization",
    local_only=True,
    description="The SentinelOne fleet, read-only: exposure to a CVE, whether an indicator "
    "has been seen, which endpoints run an application, and the state of the agents.",
    operations=(
        Operation(
            id="cve_exposure",
            summary="Which endpoints in the fleet are exposed to a given CVE.",
            ask_when="the question names a CVE id and asks who or what is exposed, affected, "
            "vulnerable or at risk in the fleet.",
            path="/web/api/v2.1/application-management/risks",
            params=_CVE,
            query={"cve": "cveId__contains"},
            const_query={"limit": "100"},
            render=Render(
                rows="data",
                line="{cveId}: {applicationName} on {endpointName} — severity {severity}",
                empty="No fleet endpoint is recorded as exposed to this CVE.",
                limit=30,
            ),
        ),
        Operation(
            id="ioc_seen",
            summary="Whether an indicator of compromise has been seen in the tenant.",
            ask_when="the question gives an indicator — a file hash, IP address, domain or "
            "URL — and asks whether it has been seen, is known, or is malicious.",
            path="/web/api/v2.1/threat-intelligence/iocs",
            params=_IOC,
            query={"indicator": "value"},
            const_query={"limit": "20"},
            render=Render(
                rows="data",
                line="{value} — {name} ({type}), source {source}",
                empty="This indicator is not in the tenant's threat-intelligence records.",
                limit=20,
            ),
        ),
        Operation(
            id="app_inventory",
            summary="Whether an application is present in the fleet, and its vendor.",
            ask_when="the question asks whether a named application or piece of software is "
            "installed, present or in use anywhere in the fleet.",
            path="/web/api/v2.1/application-management/inventory",
            params=_APP,
            query={"application": "applicationName__contains"},
            const_query={"limit": "50"},
            render=Render(
                rows="data",
                line="{applicationName} by {applicationVendor}",
                empty="No such application is in the fleet inventory.",
                limit=30,
            ),
        ),
        Operation(
            id="app_endpoints",
            summary="Which endpoints run an application, and at what version.",
            ask_when="the question asks where a named application is installed, on which "
            "hosts, or what versions of it are running.",
            path="/web/api/v2.1/application-management/inventory/endpoints",
            params=_APP,
            query={"application": "applicationName"},
            const_query={"limit": "100"},
            render=Render(
                rows="data",
                line="{endpointName}: {applicationName} {applicationVersion} ({osType})",
                empty="No endpoint is recorded as running that application.",
                limit=40,
            ),
        ),
        Operation(
            id="agents",
            summary="The agents in the fleet: how many, and their state.",
            ask_when="the question asks how many agents or endpoints there are, the size of "
            "the fleet, or the state of a named host's agent.",
            path="/web/api/v2.1/agents",
            params={
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "optional: a computer name to narrow to",
                        "maxLength": 120,
                    }
                },
            },
            query={"host": "computerName__contains"},
            const_query={"limit": "50"},
            render=Render(
                rows="data",
                line="{computerName} — agent {id}",
                empty="No agent matched.",
                limit=40,
            ),
        ),
    ),
)

BUILTIN: dict[str, Module] = {SENTINELONE.id: SENTINELONE}
