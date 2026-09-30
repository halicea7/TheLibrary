"""Identifiers a model may use only if the conversation already holds them.

A CVE id, a file hash, an IP address: a model asked to "look up those CVEs" will write
plausible ones when it can't see the real ones -- measured: "CVE-2023-1234 and
CVE-2023-5678", neither in the answer it was following up. So an identifier the model
produces must be found in the text it was given (the question, the conversation, earlier
results); otherwise what it wrote is not used.
"""

from __future__ import annotations

import re

_IDENT = re.compile(
    r"\bCVE-\d{4}-\d{4,7}\b"  # CVE ids
    r"|\bGHSA(?:-[0-9a-z]{4}){3}\b"  # GitHub advisories
    r"|\b[0-9a-f]{32}\b|\b[0-9a-f]{40}\b|\b[0-9a-f]{64}\b"  # MD5 / SHA-1 / SHA-256
    r"|\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IPv4
    re.IGNORECASE,
)


def identifiers(text: str) -> set[str]:
    return {m.group(0).lower() for m in _IDENT.finditer(text or "")}


def ungrounded(produced: str, known: str) -> set[str]:
    """Identifiers in `produced` that `known` does not contain."""
    return identifiers(produced) - identifiers(known)
