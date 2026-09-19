"""Who may talk to the library.

One rule. Requests from this machine (loopback) are the UI and are trusted, as they always
were. Requests from anywhere else must carry a bearer token from LIBRARY_API_TOKENS; with
no tokens configured, they are refused. That keeps the default -- a solo, local library --
exactly as it was, and makes opening the port a deliberate, two-step act: set a token, set
the bind address."""

from __future__ import annotations

import hmac
import ipaddress

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from library_agent.config import settings


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return host == "localhost"


def token_ok(header: str | None) -> bool:
    if not header or not header.lower().startswith("bearer "):
        return False
    given = header[7:].strip()
    # Constant-time compare against each configured token.
    return any(hmac.compare_digest(given, t) for t in settings().tokens)


class BearerOrLoopback(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if _is_loopback(request.client.host if request.client else None):
            return await call_next(request)
        if not settings().tokens:
            return JSONResponse(
                {"detail": "this library is not open to other machines (no LIBRARY_API_TOKENS)"},
                status_code=403,
            )
        if not token_ok(request.headers.get("authorization")):
            return JSONResponse(
                {"detail": "a bearer token is required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)
