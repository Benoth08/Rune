"""MCP client over stdio JSON-RPC 2.0.

This module implements the low-level transport. One :class:`MCPClient`
instance manages exactly one subprocess MCP server and provides
async-safe primitives :

- ``initialize()`` : MCP handshake (protocol version, capabilities)
- ``list_tools()`` : discover what the server exposes
- ``call_tool(name, arguments)`` : invoke a tool, get result back
- ``shutdown()`` : graceful subprocess termination

The MCP protocol spec is at https://modelcontextprotocol.io/. We
implement the **2024-11-05** revision (stable since launch). Key bits :

- Transport : stdio. Each message is a JSON-RPC 2.0 envelope on its
  own line of stdin/stdout (LSP-style framing with Content-Length
  headers is also supported but stdio is simpler and widely used).
- Sequencing : requests have an ``id``, responses match by id. We
  maintain a futures dict keyed by id so concurrent requests don't
  collide.
- Notifications : one-way messages without ``id``, used for the
  ``initialized`` handshake step and progress updates.

Concurrency
-----------
One reader task continuously drains stdout and dispatches incoming
messages to pending futures. The public methods are coroutines and
can be awaited from multiple tasks concurrently — each request gets
a unique id and a fresh future.

Error handling
--------------
We distinguish three failure categories :

- :class:`MCPTransportError` : subprocess died, JSON malformed,
  connection dropped. Means the server is unusable — caller should
  consider it disabled.
- :class:`MCPToolError` : the server responded with a JSON-RPC error
  object (code/message). Means the call failed but the server is
  alive — caller can retry or report to user.
- :class:`MCPError` : base class for everything MCP-related, useful
  for catch-all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("lythea.mcp.client")


# ── Exceptions ────────────────────────────────────────────────────────


class MCPError(Exception):
    """Base class for all MCP-related errors."""


class MCPTransportError(MCPError):
    """Subprocess died, JSON malformed, or connection unusable.

    The server is no longer reachable. Don't retry on the same client
    instance — restart the subprocess via the server manager.
    """


class MCPToolError(MCPError):
    """Server responded with a JSON-RPC error (tool failed, bad args).

    The server is alive. Caller can retry with corrected args or
    surface the error to the user.

    Attributes
    ----------
    code : int
        JSON-RPC error code (-32xxx for protocol errors, server-defined
        for tool errors).
    rpc_message : str
        Human-readable error message from the server.
    data : Any
        Optional structured payload (varies by server).
    """

    def __init__(self, code: int, rpc_message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {rpc_message}")
        self.code = code
        self.rpc_message = rpc_message
        self.data = data


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class MCPClient:
    """A single MCP server connection over stdio.

    Lifecycle
    ---------
    1. Construct with command + args (and optional env extras)
    2. Call ``await start()`` to spawn the subprocess and handshake
    3. Use ``list_tools()`` / ``call_tool()`` freely
    4. Call ``await shutdown()`` for graceful cleanup

    The class is **not** safe to share across event loops, but is
    safe for concurrent access within one event loop.
    """

    command: str
    args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)
    name: str = ""  # used in logs and errors

    # Internal state, not part of construction
    _process: asyncio.subprocess.Process | None = None
    _reader_task: asyncio.Task | None = None
    _pending: dict[int, asyncio.Future] = field(default_factory=dict)
    _next_id: int = 1
    _server_info: dict = field(default_factory=dict)
    _started: bool = False
    _shutting_down: bool = False
    _stderr_buffer: list[str] = field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        """Whether the subprocess is currently running."""
        return (
            self._process is not None
            and self._process.returncode is None
            and not self._shutting_down
        )

    async def start(self, *, timeout: float = 10.0) -> None:
        """Spawn the subprocess and perform the MCP handshake.

        Raises
        ------
        MCPTransportError
            If the executable can't be found, the subprocess crashes
            during startup, or the handshake times out.
        """
        if self._started:
            raise MCPTransportError(f"Client {self.name!r} already started")

        # Resolve the command. If it's "npx" or "node", look it up on
        # PATH first to get an absolute path (clearer error messages).
        resolved = shutil.which(self.command) or self.command

        # Build the subprocess environment. We keep most of the parent
        # env (so $HOME, $PATH, etc. work) but layer extras on top.
        env = os.environ.copy()
        env.update(self.extra_env)

        log.info(
            "MCP[%s] spawning: %s %s",
            self.name, resolved, " ".join(self.args),
        )

        try:
            self._process = await asyncio.create_subprocess_exec(
                resolved,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as exc:
            raise MCPTransportError(
                f"MCP[{self.name}] executable not found: {resolved!r}"
            ) from exc
        except Exception as exc:
            raise MCPTransportError(
                f"MCP[{self.name}] failed to spawn: {exc}"
            ) from exc

        # Start background readers for stdout (JSON-RPC frames) and
        # stderr (server logs, kept in buffer for diagnostics).
        self._reader_task = asyncio.create_task(
            self._read_stdout_loop(),
            name=f"mcp-reader-{self.name}",
        )
        asyncio.create_task(
            self._read_stderr_loop(),
            name=f"mcp-stderr-{self.name}",
        )

        # MCP handshake
        try:
            await asyncio.wait_for(self._handshake(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            await self._force_kill()
            raise MCPTransportError(
                f"MCP[{self.name}] handshake timeout after {timeout}s. "
                f"stderr tail: {self._tail_stderr(500)!r}"
            ) from exc
        except MCPError:
            await self._force_kill()
            raise

        self._started = True
        log.info(
            "MCP[%s] ready — server_info=%s",
            self.name, self._server_info,
        )

    async def _handshake(self) -> None:
        """Send ``initialize`` then ``notifications/initialized``."""
        result = await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": False},
                    "sampling": {},
                },
                "clientInfo": {
                    "name": "lythea",
                    "version": "6.0.0",
                },
            },
        )
        self._server_info = result.get("serverInfo", {})

        # Send the "initialized" notification (no response expected)
        await self._send_raw({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })

    async def list_tools(self) -> list[dict]:
        """Return the list of tools exposed by the server.

        Each item has at least ``name``, ``description``, and an
        ``inputSchema`` (JSON Schema). The exact shape depends on the
        MCP server but follows the spec.
        """
        result = await self._request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(
        self,
        name: str,
        arguments: dict | None = None,
        *,
        timeout: float = 30.0,
    ) -> dict:
        """Invoke a tool and return the raw result.

        The result follows the MCP spec : a dict with ``content``
        (list of content blocks : text, image, resource) and an
        optional ``isError`` flag.

        Parameters
        ----------
        name : str
            Tool name as exposed by ``list_tools()``.
        arguments : dict, optional
            Tool-specific arguments (matches the tool's inputSchema).
        timeout : float
            Max time for the call. MCP tools can be slow (e.g. a web
            fetch), so this is generous by default.

        Raises
        ------
        MCPToolError
            If the server returns an error response.
        MCPTransportError
            If the subprocess dies or the call times out.
        """
        try:
            result = await asyncio.wait_for(
                self._request("tools/call", {
                    "name": name,
                    "arguments": arguments or {},
                }),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise MCPTransportError(
                f"MCP[{self.name}] tool {name!r} timeout after {timeout}s"
            ) from exc
        return result

    async def shutdown(self) -> None:
        """Graceful subprocess termination.

        Tries SIGTERM first, falls back to SIGKILL after 2 seconds.
        Safe to call multiple times.
        """
        if self._shutting_down or self._process is None:
            return
        self._shutting_down = True
        log.info("MCP[%s] shutting down", self.name)

        # Fail any pending requests
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(
                    MCPTransportError(f"MCP[{self.name}] shutting down")
                )
        self._pending.clear()

        # Terminate the subprocess
        if self._process.returncode is None:
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    await self._force_kill()
            except ProcessLookupError:
                pass  # already gone

        # Cancel reader task
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _force_kill(self) -> None:
        if self._process is None:
            return
        try:
            self._process.kill()
            await self._process.wait()
        except (ProcessLookupError, Exception):
            pass

    # ── JSON-RPC internals ───────────────────────────────────────────

    async def _request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and await the response."""
        if not self.is_alive and method != "initialize":
            raise MCPTransportError(
                f"MCP[{self.name}] not alive (cannot call {method!r})"
            )

        req_id = self._next_id
        self._next_id += 1

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut

        try:
            await self._send_raw({
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            })
            response = await fut
        finally:
            self._pending.pop(req_id, None)

        if "error" in response:
            err = response["error"]
            raise MCPToolError(
                code=err.get("code", -32000),
                rpc_message=err.get("message", "unknown error"),
                data=err.get("data"),
            )
        return response.get("result", {})

    async def _send_raw(self, payload: dict) -> None:
        """Encode payload as JSON + newline, write to stdin."""
        if self._process is None or self._process.stdin is None:
            raise MCPTransportError(
                f"MCP[{self.name}] stdin not available"
            )
        try:
            data = (json.dumps(payload) + "\n").encode("utf-8")
            self._process.stdin.write(data)
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise MCPTransportError(
                f"MCP[{self.name}] connection lost on write"
            ) from exc

    async def _read_stdout_loop(self) -> None:
        """Continuously drain stdout, dispatch responses to futures."""
        if self._process is None or self._process.stdout is None:
            return
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break  # EOF, subprocess died
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    log.warning(
                        "MCP[%s] non-JSON line on stdout: %r",
                        self.name, line[:200],
                    )
                    continue
                self._dispatch_message(msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("MCP[%s] reader loop crashed", self.name)
        finally:
            # Fail any remaining pending requests
            err = MCPTransportError(
                f"MCP[{self.name}] subprocess closed. "
                f"stderr: {self._tail_stderr(500)!r}"
            )
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(err)
            self._pending.clear()

    async def _read_stderr_loop(self) -> None:
        """Drain stderr into a ring buffer for diagnostics."""
        if self._process is None or self._process.stderr is None:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                self._stderr_buffer.append(decoded)
                # Keep buffer bounded
                if len(self._stderr_buffer) > 200:
                    self._stderr_buffer = self._stderr_buffer[-200:]
                # Echo at DEBUG so it's not noisy by default
                log.debug("MCP[%s] stderr: %s", self.name, decoded)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.debug("MCP[%s] stderr reader stopped", self.name, exc_info=True)

    def _dispatch_message(self, msg: dict) -> None:
        """Route an incoming JSON-RPC message."""
        if "id" in msg and ("result" in msg or "error" in msg):
            # Response to a request
            fut = self._pending.get(msg["id"])
            if fut is not None and not fut.done():
                fut.set_result(msg)
            else:
                log.warning(
                    "MCP[%s] response for unknown id %r", self.name, msg["id"],
                )
        elif "method" in msg:
            # Notification or server-initiated request. We don't
            # implement reverse calls (the server asking us to do
            # things) yet — log and ignore.
            method = msg["method"]
            if method.startswith("notifications/"):
                log.debug("MCP[%s] notification: %s", self.name, method)
            else:
                log.debug(
                    "MCP[%s] server-initiated request ignored: %s",
                    self.name, method,
                )
        else:
            log.warning("MCP[%s] malformed message: %r", self.name, msg)

    def _tail_stderr(self, max_chars: int = 500) -> str:
        """Last bytes of stderr, useful for error messages."""
        joined = "\n".join(self._stderr_buffer)
        if len(joined) > max_chars:
            return "..." + joined[-max_chars:]
        return joined
