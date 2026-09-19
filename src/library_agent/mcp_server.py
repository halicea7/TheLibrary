"""The library as tools for an agent (MCP).

Four tools -- what is here, find passages, ask a question, read a volume -- served over
stdio (Claude Desktop, Claude Code, most agent frameworks launch it that way) or over
HTTP for an agent on another machine. It is a thin client of the library's own JSON API,
so one Ollama, one job at a time, and one set of rules about who may ask stay in one
place: the agent gets exactly what a person at the desk gets, with the same verified
citations, and nothing more.

    ./library mcp                       # stdio
    ./library mcp --http 8078           # streamable HTTP on :8078
    LIBRARY_URL / LIBRARY_TOKEN         # where the library is, and the bearer if not local
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

BASE = os.environ.get("LIBRARY_URL", "http://127.0.0.1:8077").rstrip("/")
TOKEN = os.environ.get("LIBRARY_TOKEN", "")

server = MCPServer(
    "the-library",
    instructions=(
        "A personal research library that has read its documents. Use list_shelves to see "
        "what it holds and which rooms (cartridges) exist; search_library for passages; "
        "ask_library for an answer with verified citations -- every [n] in the answer maps to "
        "a citation with the document, page and section. Pass room to stay inside one "
        "collection. Anything in an answer without a citation is the librarian's own reasoning."
    ),
)


def _client() -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    return httpx.AsyncClient(base_url=BASE, headers=headers, timeout=600)


@server.tool(
    description="What the library holds: top shelves with sub-shelves and counts, and the rooms (cartridges) you can ask by name."
)
async def list_shelves() -> dict[str, Any]:
    async with _client() as c:
        r = await c.get("/api/v1/shelves")
        r.raise_for_status()
        return r.json()


@server.tool(
    description="Find passages. Returns ranked excerpts with document, page and section. Use room to search one cartridge, subjects (comma-separated shelf names) to stay within shelves."
)
async def search_library(
    query: str, room: str | None = None, subjects: str | None = None, limit: int = 8
) -> dict[str, Any]:
    async with _client() as c:
        r = await c.get(
            "/api/v1/search",
            params={"q": query, "room": room, "subjects": subjects, "limit": limit},
        )
        r.raise_for_status()
        return r.json()


@server.tool(
    description="Ask the library a question and get an answer with verified citations. room: a cartridge name to stay inside. conversation_id: pass back to continue a thread. stance: opinionated | contrarian | charitable | cynical | optimistic."
)
async def ask_library(
    question: str,
    room: str | None = None,
    subjects: list[str] | None = None,
    conversation_id: str | None = None,
    stance: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    body = {
        "question": question,
        "room": room,
        "subjects": subjects or [],
        "conversation_id": conversation_id,
        "stance": stance,
        "model": model,
    }
    async with _client() as c:
        r = await c.post("/api/v1/ask", json=body)
        if r.status_code >= 400:
            return {"error": r.json().get("detail", r.text)[:500]}
        return r.json()


@server.tool(
    description="Read a volume: its sections, passages, and what the library wrote about them. document_id comes from search or citations."
)
async def read_volume(document_id: str, max_chars: int = 20000) -> dict[str, Any]:
    async with _client() as c:
        r = await c.get(f"/api/v1/volumes/{document_id}")
        r.raise_for_status()
        d = r.json()
    # Keep it readable for a model: sections with their passages, trimmed to a budget.
    out, used = [], 0
    for s in d.get("sections", []):
        block = {
            "title": s.get("title"),
            "summary": s.get("summary"),
            "page": s.get("page_start"),
            "passages": [],
        }
        for p in s.get("passages", []):
            txt = p["text"]
            if used + len(txt) > max_chars:
                txt = txt[: max(0, max_chars - used)]
            if txt:
                block["passages"].append(
                    {"page": p.get("page"), "text": txt, "note": p.get("reflection")}
                )
                used += len(txt)
            if used >= max_chars:
                break
        out.append(block)
        if used >= max_chars:
            break
    return {
        "id": d["id"],
        "title": d["title"],
        "cartridge": d.get("cartridge"),
        "sections": out,
        "truncated": used >= max_chars,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="library mcp")
    ap.add_argument(
        "--http", type=int, default=None, help="serve streamable HTTP on this port instead of stdio"
    )
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    if a.http:
        server.settings.host = a.host
        server.settings.port = a.http
        server.run("streamable-http")
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
