"""MCP server orchestrator.

Reads the MCP server config from settings, spawns each declared
server at boot, holds the registry of available tools (server →
tool name → schema), and dispatches calls to the right
:class:`MCPClient`.

Public API
----------
Single class :class:`MCPServerManager`. Created in :mod:`rune.boot`,
stored on the FastAPI app as ``app.mcp_manager``. Used by hippocampe
and routes via ``app.mcp_manager.<method>``.

Default servers
---------------
V6.0.0 ships with three servers wired by default :

- ``filesystem`` — read/write inside the Lythéa sandbox dir.
- ``github`` — read-only access (PAT via env var).
- ``youtube`` — transcript fetcher, no auth.

Each can be disabled individually via settings flags.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .client import MCPClient, MCPError
from .prerequisites import check_node, log_status_or_instructions

log = logging.getLogger("rune.mcp.server_manager")


@dataclass(frozen=True)
class MCPToolInfo:
    """Public view of a tool exposed by an MCP server.

    Used by the router to pick the right tool and by the UI to
    surface what Lythéa can do.

    Attributes
    ----------
    server : str
        Server name (e.g. "filesystem").
    name : str
        Tool name within the server (e.g. "read_file").
    description : str
        Human description from the server's tools/list response.
    input_schema : dict
        JSON Schema for the arguments. Used by the LLM to format
        proper tool calls.
    """

    server: str
    name: str
    description: str
    input_schema: dict

    @property
    def qualified_name(self) -> str:
        """Globally unique identifier across all servers."""
        return f"{self.server}.{self.name}"


@dataclass
class _ServerConfig:
    """Internal config for one MCP server, post-resolution."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


class MCPServerManager:
    """Orchestrate multiple MCP servers + tools registry.

    Lifecycle
    ---------
    1. ``__init__`` : parse config from settings, build _ServerConfig
       list (cheap, sync).
    2. ``await start_all()`` : spawn enabled servers in parallel,
       run handshakes, populate tool registry.
    3. ``list_tools()`` / ``call_tool()`` during runtime.
    4. ``await shutdown_all()`` on app shutdown.

    Notes
    -----
    Servers that fail to start are logged and disabled — they don't
    take down the manager. ``available_servers()`` returns only the
    ones that came up successfully.
    """

    def __init__(self, sandbox_dir: Path) -> None:
        """Initialise from Lythéa settings + sandbox dir.

        Parameters
        ----------
        sandbox_dir : Path
            Absolute path to the workspace sandbox. Used as the
            filesystem MCP's scope. Created if it doesn't exist.
        """
        self.sandbox_dir = Path(sandbox_dir).expanduser().resolve()
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, MCPToolInfo] = {}  # qualified_name → info
        self._configs: list[_ServerConfig] = []
        self._started: bool = False
        self._node_available: bool = False

        self._build_default_config()

    def _build_default_config(self) -> None:
        """Build the default V6.0.0 server list.

        Servers can be customised by editing settings later. For
        now we hardcode the three defaults : filesystem, github,
        youtube.
        """
        from rune.settings import get_settings
        s = get_settings()

        # ── Filesystem MCP ───────────────────────────────────────
        # Scoped to the sandbox dir only. Lythéa cannot escape this
        # directory via the filesystem server, by design.
        if getattr(s, "mcp_filesystem_enabled", True):
            self._configs.append(_ServerConfig(
                name="filesystem",
                command="npx",
                args=[
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    str(self.sandbox_dir),
                ],
            ))

        # ── GitHub MCP (read-only) ──────────────────────────────
        # The PAT is read from env var. If absent, the server still
        # starts but can only access public repos. We log a hint.
        if getattr(s, "mcp_github_enabled", True):
            pat = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
            self._configs.append(_ServerConfig(
                name="github",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": pat} if pat else {},
            ))
            if not pat:
                log.info(
                    "MCP[github] no PAT set (env GITHUB_PERSONAL_ACCESS_TOKEN) "
                    "— public repos only"
                )

        # ── YouTube Transcript MCP ──────────────────────────────
        # No API key needed for transcripts (publicly available).
        if getattr(s, "mcp_youtube_enabled", True):
            self._configs.append(_ServerConfig(
                name="youtube",
                command="npx",
                args=["-y", "@sinco-lab/mcp-youtube-transcript"],
            ))

    async def start_all(self) -> None:
        """Spawn all enabled servers in parallel.

        Blocking call : returns when all handshakes are done (or have
        failed). Servers that fail are excluded from the tool registry
        but the manager remains usable for the rest.
        """
        if self._started:
            log.warning("MCPServerManager already started — skipping")
            return

        # Check Node.js first. If absent, nothing will work.
        node_status = check_node()
        log_status_or_instructions(node_status)
        self._node_available = node_status.available
        if not self._node_available:
            log.warning(
                "MCP servers disabled — Node.js not available. "
                "See logs above for install instructions."
            )
            self._started = True  # mark as "started" (with 0 servers)
            return

        # Launch each server in parallel — failures don't propagate
        results = await asyncio.gather(
            *[self._start_one(cfg) for cfg in self._configs if cfg.enabled],
            return_exceptions=True,
        )

        n_ok = sum(1 for r in results if r is True)
        n_fail = sum(1 for r in results if r is not True)
        log.info(
            "MCPServerManager: %d/%d servers ready, %d tools available",
            n_ok, len(self._configs), len(self._tools),
        )
        if n_fail:
            log.warning(
                "MCPServerManager: %d servers failed — see preceding logs",
                n_fail,
            )

        self._started = True

    async def _start_one(self, cfg: _ServerConfig) -> bool:
        """Spawn one server, list its tools, register them."""
        client = MCPClient(
            command=cfg.command,
            args=list(cfg.args),
            extra_env=dict(cfg.env),
            name=cfg.name,
        )
        try:
            await client.start(timeout=30.0)  # npx may need to download
        except MCPError as exc:
            log.warning("MCP[%s] failed to start: %s", cfg.name, exc)
            return False
        except Exception:
            log.exception("MCP[%s] unexpected start failure", cfg.name)
            return False

        # List tools
        try:
            tools = await client.list_tools()
        except MCPError as exc:
            log.warning("MCP[%s] tools/list failed: %s", cfg.name, exc)
            await client.shutdown()
            return False

        self._clients[cfg.name] = client
        for tool in tools:
            info = MCPToolInfo(
                server=cfg.name,
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
            )
            if info.name:
                self._tools[info.qualified_name] = info

        log.info(
            "MCP[%s] ready with %d tools: %s",
            cfg.name, len(tools),
            ", ".join(t.get("name", "?") for t in tools[:10]),
        )
        return True

    async def shutdown_all(self) -> None:
        """Terminate every running server gracefully."""
        await asyncio.gather(
            *[c.shutdown() for c in self._clients.values()],
            return_exceptions=True,
        )
        self._clients.clear()
        self._tools.clear()
        self._started = False

    # ── Public API ───────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Whether at least one MCP server is operational."""
        return any(c.is_alive for c in self._clients.values())

    def available_servers(self) -> list[str]:
        """Names of servers currently running."""
        return [n for n, c in self._clients.items() if c.is_alive]

    def list_tools(self, server: str | None = None) -> list[MCPToolInfo]:
        """Return all tools, optionally filtered by server name."""
        if server is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if t.server == server]

    def get_tool(self, qualified_name: str) -> MCPToolInfo | None:
        """Look up a tool by its ``server.tool`` qualified name."""
        return self._tools.get(qualified_name)

    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict | None = None,
        *,
        timeout: float = 30.0,
    ) -> dict:
        """Invoke a tool on a specific server.

        Returns the raw MCP result (dict with ``content``, optional
        ``isError``).

        Raises
        ------
        KeyError
            If the server isn't running.
        MCPError
            Propagated from the client.
        """
        client = self._clients.get(server)
        if client is None or not client.is_alive:
            raise KeyError(f"MCP server {server!r} not available")
        return await client.call_tool(tool, arguments, timeout=timeout)

    def snapshot(self) -> dict:
        """Diagnostic dict for the debug endpoint."""
        return {
            "node_available": self._node_available,
            "started": self._started,
            "servers": {
                name: {
                    "alive": client.is_alive,
                    "n_tools": sum(
                        1 for t in self._tools.values() if t.server == name
                    ),
                }
                for name, client in self._clients.items()
            },
            "n_tools_total": len(self._tools),
            "sandbox_dir": str(self.sandbox_dir),
        }
