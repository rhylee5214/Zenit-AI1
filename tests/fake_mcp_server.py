"""A tiny real MCP stdio server that imitates the Roblox Studio MCP proxy.

Run: python tests/fake_mcp_server.py
Tools: list_roblox_studios, execute_luau, fail_tool, crash, crash_soon. Used by tests/test_roblox.py
to check that MCPBridge really talks MCP over stdio.
"""

from __future__ import annotations

import json

try:                                          # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                           # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

server = _Server("FakeRobloxStudio")


@server.tool()
def list_roblox_studios() -> str:
    """List all connected Roblox Studio instances"""
    return json.dumps({"studios": [{"id": "studio-test1", "name": "Baseplate"},
                                   {"id": "studio-test2", "name": "Obby"}]})


@server.tool()
def execute_luau(code: str, datamodel_type: str, studio_id: str) -> str:
    """Executes Luau code in Roblox Studio."""
    return f"ran {len(code)} chars in {datamodel_type} on {studio_id}"


@server.tool()
def fail_tool(studio_id: str) -> str:
    """Always fails."""
    raise ValueError("Studio says no")


@server.tool()
def crash() -> str:
    """Kills this server process (to test reconnects)."""
    import os
    os._exit(3)


@server.tool()
def crash_soon() -> str:
    """Answers, then the server process exits while idle (to test dead-session detection)."""
    import os
    import threading
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return "bye"


if __name__ == "__main__":
    server.run("stdio")
