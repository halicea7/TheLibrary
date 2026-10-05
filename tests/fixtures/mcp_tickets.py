"""A tiny MCP server for the tests: one read-only tool, one destructive one, one that
reports what environment it was given, one that says which process answered (so a kept
session can be told from a fresh one), one with a declared output, and two resources."""

import os

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel


class Ticket(BaseModel):
    id: int
    title: str
    status: str
    opened: str
    tags: list[str]


class TicketList(BaseModel):
    total: int
    tickets: list[Ticket]


app = MCPServer("tickets")


@app.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_tickets(status: str = "open", limit: int = 5) -> dict:
    """Tickets by status."""
    items = [{"id": i, "title": f"ticket {i}", "status": status} for i in range(1, 4)]
    return {"items": items[:limit], "key_seen": os.environ.get("TICKETS_KEY", "")}


@app.tool(annotations=ToolAnnotations(readOnlyHint=True))
def env_names() -> str:
    """The environment variable names this server was started with."""
    return "\n".join(sorted(os.environ))


@app.tool(annotations=ToolAnnotations(readOnlyHint=True))
def whoami() -> str:
    """The id of the process answering."""
    return str(os.getpid())


@app.tool(annotations=ToolAnnotations(readOnlyHint=True))
def search_tickets(q: str) -> TicketList:
    """Tickets matching a phrase."""
    rows = [
        Ticket(
            id=7, title=f"{q} fails", status="open", opened="2026-09-30T10:00:00Z", tags=["vpn"]
        ),
        Ticket(id=9, title=f"{q} slow", status="closed", opened="2026-09-29T08:00:00Z", tags=[]),
    ]
    return TicketList(total=len(rows), tickets=rows)


@app.resource("tickets://guide", description="How the desk triages tickets.")
def guide() -> str:
    return "Triage first.\nThen assign.\n\nClose only when the reporter agrees."


@app.resource("tickets://ticket/{id}", description="One ticket's notes.")
def ticket_notes(id: str) -> str:
    return f"Notes for ticket {id}."


@app.tool(annotations=ToolAnnotations(destructiveHint=True))
def close_ticket(id: int) -> str:
    """Close a ticket."""
    return "closed"


if __name__ == "__main__":
    app.run()
