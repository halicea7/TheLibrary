"""What a module is: a line to an API, described so the librarian can consult it during a
question without ever writing a URL or code.

A module is not a shelf of documents; it is a small, curated set of read-only operations.
Each operation declares the parameters it takes (a JSON schema the picker fills), when it
applies (a sentence for the planner), the GET it maps to (a path template and a mapping of
declared parameters to query keys), and how the JSON that comes back becomes passage text
(a row selector and a line format). Everything the model touches is a declared field; the
transport is built from the manifest, not from anything the model emits.

Only the base URL and the auth token are the operator's to set, and the token never lives
here — it is held beside the provider keys, 0600, and never shown."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Render:
    """How one JSON response becomes passage text: the array to walk, the fields to pull,
    and the line each row prints as. No model call -- a citation must trace to what came
    back."""

    rows: str  # dotted path to the array of rows, e.g. "data"
    line: str  # a format string over a row's fields, e.g. "{cveId}: {applicationName} on {endpointName} ({severity})"
    empty: str = "nothing matched"  # printed when the array is empty
    limit: int = 20  # rows rendered at most


@dataclass(frozen=True)
class Operation:
    id: str
    summary: str  # what it answers, one line
    ask_when: str  # for the picker: when this operation applies
    path: str  # a GET path template, e.g. "/web/api/v2.1/application-management/risks"
    params: dict[str, Any]  # JSON schema for the operation's parameters
    query: dict[str, str]  # declared-parameter -> query-key, e.g. {"cve": "cveId__contains"}
    render: Render
    const_query: dict[str, str] = field(
        default_factory=dict
    )  # fixed query keys, e.g. {"limit": "50"}


@dataclass(frozen=True)
class Module:
    id: str
    name: str
    kind: str  # a short noun for the object/label, e.g. "endpoint security"
    colour: str  # the live-result pigment on this module's citations
    auth_scheme: str  # "apitoken" or "bearer"
    auth_header: str  # the header the token goes in, e.g. "Authorization"
    operations: tuple[Operation, ...]
    local_only: bool = True  # refuse to consult when chat is on a remote provider
    description: str = ""

    def op(self, op_id: str) -> Operation | None:
        return next((o for o in self.operations if o.id == op_id), None)

    def token_prefix(self) -> str:
        return {"apitoken": "ApiToken ", "bearer": "Bearer "}.get(self.auth_scheme, "")
