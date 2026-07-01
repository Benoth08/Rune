"""Lythéa MCP client — Model Context Protocol integration.

Architecture
------------
Lythéa speaks JSON-RPC 2.0 over stdio to MCP servers (subprocess).
Three core components :

- :mod:`prerequisites` : detects Node.js, gives install instructions
  if missing. Called at boot before any server start.
- :mod:`client` : low-level JSON-RPC client. Manages one subprocess
  connection, handshake, tools/list, tools/call.
- :mod:`server_manager` : top-level orchestrator. Reads config,
  spawns the declared MCP servers at boot, holds the registry
  of available tools, dispatches calls to the right client.

Public API for the rest of Lythéa
---------------------------------
The :class:`MCPServerManager` singleton is created at boot
(:func:`lythea.boot.init_mcp`) and stored on the FastAPI app
(``app.mcp_manager``). Callers (hippocampe, routes) use::

    tools = app.mcp_manager.list_tools()
    result = await app.mcp_manager.call_tool(
        server="filesystem",
        tool="read_file",
        arguments={"path": "/workspace/.lythea/sandbox/data.csv"},
    )

Errors are wrapped in :class:`MCPError` (or subclasses) so callers
can distinguish transport failures, server crashes, and tool errors.
"""

from .client import MCPClient, MCPError, MCPTransportError, MCPToolError
from .prerequisites import check_node, NODE_INSTALL_INSTRUCTIONS
from .server_manager import MCPServerManager, MCPToolInfo

__all__ = [
    "MCPClient",
    "MCPError",
    "MCPTransportError",
    "MCPToolError",
    "MCPServerManager",
    "MCPToolInfo",
    "check_node",
    "NODE_INSTALL_INSTRUCTIONS",
]
