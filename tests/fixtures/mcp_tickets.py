"""A tiny MCP server for the tests: one read-only tool, one destructive one, one that
reports what environment it was given."""

import os

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

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


@app.tool(annotations=ToolAnnotations(destructiveHint=True))
def close_ticket(id: int) -> str:
    """Close a ticket."""
    return "closed"


if __name__ == "__main__":
    app.run()
