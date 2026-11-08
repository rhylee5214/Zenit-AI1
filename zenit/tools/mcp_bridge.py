"""A synchronous, thread-safe client for one MCP server over stdio.

The MCP Python SDK is async (anyio). ZENIT's API calls tools from worker
threads, so MCPBridge owns a background thread with its own asyncio loop and
keeps ONE persistent ClientSession open there. Sync methods submit
coroutines to that loop and wait for the result.

The session lives inside a single long-running "owner" task, because anyio
cancel scopes (used by stdio_client and ClientSession) must be entered and
exited by the same task. close() signals that task, which exits the contexts
and terminates the server process tree.

Works with mcp 1.x (camelCase fields) and 2.x (snake_case fields).
On Windows the loop is a ProactorEventLoop, which subprocess pipes need.
"""

from __future__ import annotations

import asyncio
import builtins
import concurrent.futures
import json
import os
import sys
import threading
from typing import Any


class MCPError(RuntimeError):
    pass


class MCPConnectError(MCPError):
    """The server process could not be started or did not answer initialize."""


def _root_cause(err: BaseException) -> BaseException:
    group = getattr(builtins, "BaseExceptionGroup", None)     # Python 3.11+
    while group is not None and isinstance(err, group) and getattr(err, "exceptions", None):
        err = err.exceptions[0]
    return err


def mcp_available() -> bool:
    try:
        import mcp  # noqa: F401
        from mcp.client.stdio import stdio_client  # noqa: F401
        return True
    except Exception:
        return False


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def flatten_result(result: Any) -> tuple[bool, str]:
    """CallToolResult -> (ok, text). Text blocks are joined; images and
    other binary blocks become short placeholders."""
    is_error = bool(_field(result, "is_error", "isError", default=False))
    parts: list[str] = []
    for block in _field(result, "content", default=None) or []:
        kind = _field(block, "type", default="")
        if kind == "text":
            parts.append(str(_field(block, "text", default="")))
        elif kind == "image":
            parts.append("[image omitted]")
        elif kind == "audio":
            parts.append("[audio omitted]")
        elif kind == "resource":
            res = _field(block, "resource", default=None)
            text = _field(res, "text", default=None) if res is not None else None
            parts.append(str(text) if text is not None else "[resource omitted]")
        elif kind == "resource_link":
            parts.append(f"[resource link: {_field(block, 'uri', default='')}]")
        else:
            parts.append(f"[{kind or 'content'} omitted]")
    text = "\n".join(p for p in parts if p)
    if not text:
        structured = _field(result, "structured_content", "structuredContent", default=None)
        if structured is not None:
            text = json.dumps(structured, ensure_ascii=False)
    if not text and _field(result, "content", default=None) is None:
        # Not a CallToolResult (e.g. an input-required result in mcp 2.x).
        return False, f"unexpected MCP result: {type(result).__name__}"
    return not is_error, text


_SEND_FAILED = {"ClosedResourceError", "BrokenResourceError", "BrokenPipeError", "MCPSessionGone"}


def _send_failed(err: BaseException) -> bool:
    """The request never reached the server (its pipe was already closed)."""
    return type(err).__name__ in _SEND_FAILED or isinstance(err, BrokenPipeError)


def _connection_lost(err: BaseException) -> bool:
    """The connection died, possibly after the server received the request."""
    if _send_failed(err) or isinstance(err, (ConnectionError, EOFError)):
        return True
    if type(err).__name__ in ("EndOfStream", "ConnectionResetError"):
        return True
    msg = str(err).lower()
    return "connection closed" in msg or ("closed" in msg and "session" in msg)


class MCPSessionGone(MCPError):
    pass


def _session_dead(session: Any) -> bool:
    """True when the session's transport has already closed (the server
    process exited while idle). Best effort: mcp 2.x keeps the receive loop
    state on the session's JSON-RPC dispatcher; unknown layouts report alive."""
    if session is None:
        return True
    for owner in (getattr(session, "_dispatcher", None), session):
        if owner is not None and getattr(owner, "_closed", False) is True:
            return True
    return False


WATCH_INTERVAL = 0.25          # seconds between transport liveness checks while idle


class MCPBridge:
    def __init__(self, command: str, args: list[str] | None = None, env: dict[str, str] | None = None,
                 cwd: str | None = None, log_path: str | None = None,
                 connect_timeout: float = 30.0) -> None:
        self.command = command
        self.connect_timeout = connect_timeout
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd
        self.log_path = log_path
        self.server_info: dict = {}
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session = None
        self._owner: asyncio.Future | None = None       # concurrent future of the owner task
        self._stop: asyncio.Event | None = None
        self._tools: list[dict] | None = None
        self._errlog = None

    # ------------------------------------------------------------ loop thread
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and self._thread is not None and self._thread.is_alive():
            return self._loop
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)
            try:
                loop.run_forever()
            finally:
                try:
                    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                    for t in pending:
                        t.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.wait(pending, timeout=3))
                except Exception:
                    pass
                loop.close()

        th = threading.Thread(target=run, name="zenit-mcp-bridge", daemon=True)
        th.start()
        ready.wait(5)
        self._loop, self._thread = loop, th
        return loop

    def _submit(self, coro) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())

    # ----------------------------------------------------------- owner task
    async def _owner_task(self, ready: asyncio.Future) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client

        env = {**get_default_environment(), **self.env} if self.env else None
        params = StdioServerParameters(command=self.command, args=self.args, env=env, cwd=self.cwd)
        if self._errlog is None:
            self._errlog = open(self.log_path or os.devnull, "a", encoding="utf-8")
        self._stop = asyncio.Event()
        try:
            async with stdio_client(params, errlog=self._errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    info = _field(init, "server_info", "serverInfo", default=None)
                    self.server_info = {"name": _field(info, "name", default=None),
                                        "version": _field(info, "version", default=None)} if info else {}
                    self._session = session
                    if not ready.done():
                        ready.set_result(True)
                    # Wait for close(); leave early if the server process dies while idle,
                    # so `connected` turns False and the next call starts a fresh server
                    # instead of sending its request into a dead session.
                    while not self._stop.is_set() and not _session_dead(session):
                        await asyncio.sleep(WATCH_INTERVAL)
        except BaseException as err:
            if not ready.done():
                ready.set_exception(err if isinstance(err, Exception) else MCPError(repr(err)))
            if isinstance(err, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
        finally:
            self._session = None
            self._tools = None

    async def _start(self) -> None:
        loop = asyncio.get_running_loop()
        ready: asyncio.Future = loop.create_future()
        self._owner = loop.create_task(self._owner_task(ready))
        await ready

    async def _shutdown(self) -> None:
        if self._stop is not None:
            self._stop.set()
        owner = self._owner
        if owner is not None and not owner.done():
            try:
                await asyncio.wait_for(asyncio.shield(owner), timeout=5)
            except BaseException:
                owner.cancel()
                try:
                    await asyncio.wait_for(owner, timeout=3)
                except BaseException:
                    pass
        self._owner = None
        self._session = None

    # --------------------------------------------------------------- public
    @property
    def connected(self) -> bool:
        session = self._session
        return (session is not None and self._owner is not None and not self._owner.done()
                and not _session_dead(session))

    def connect(self, timeout: float = 30.0) -> None:
        """Start the server process and open the session (no-op if open)."""
        with self._lock:
            if self.connected:
                return
            if not mcp_available():
                raise MCPConnectError("the 'mcp' package is not installed (pip install -e \".[roblox]\")")
            fut = self._submit(self._shutdown())     # clear any dead session first
            fut.result(10)
            fut = self._submit(self._start())
            try:
                fut.result(timeout)
            except concurrent.futures.TimeoutError:
                fut.cancel()
                self._submit(self._shutdown())
                raise MCPConnectError(f"MCP server did not start within {timeout:.0f}s")
            except Exception as err:
                self._submit(self._shutdown())
                cause = _root_cause(err)
                raise MCPConnectError(f"could not start MCP server: {type(cause).__name__}: {cause}") from err

    def close(self) -> None:
        with self._lock:
            loop, th = self._loop, self._thread
            if loop is None or th is None:
                return
            if th.is_alive():
                try:
                    asyncio.run_coroutine_threadsafe(self._shutdown(), loop).result(10)
                except Exception:
                    pass
                loop.call_soon_threadsafe(loop.stop)
                th.join(5)
            self._loop = self._thread = None
            if self._errlog is not None:
                try:
                    self._errlog.close()
                except Exception:
                    pass
                self._errlog = None

    def _run(self, make_coro, timeout: float):
        fut = self._submit(make_coro())
        try:
            return fut.result(timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise

    def list_tools(self, timeout: float = 30.0) -> list[dict]:
        def once():
            self.connect(timeout)
            session = self._session
            if session is None:
                raise MCPSessionGone("MCP session closed")

            async def go():
                out, cursor = [], None
                while True:
                    if cursor:
                        try:
                            from mcp.types import PaginatedRequestParams
                            res = await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
                        except Exception:
                            res = await session.list_tools(cursor=cursor)
                    else:
                        res = await session.list_tools()
                    for t in _field(res, "tools", default=[]) or []:
                        out.append({"name": _field(t, "name"), "description": _field(t, "description") or "",
                                    "input_schema": _field(t, "input_schema", "inputSchema", default={}) or {}})
                    cursor = _field(res, "next_cursor", "nextCursor", default=None)
                    if not cursor:
                        return out
            return self._run(go, timeout)

        with self._lock:
            if self._tools is not None and self.connected:
                return list(self._tools)
        tools = self._with_reconnect(once, idempotent=True)
        self._tools = tools
        return list(tools)

    def call_tool(self, name: str, arguments: dict | None = None, timeout: float = 120.0) -> tuple[bool, str]:
        """Run one tool. Returns (ok, text); never raises for tool-level
        failures, timeouts or a dead connection."""
        def once():
            self.connect(self.connect_timeout)
            session = self._session
            if session is None:
                raise MCPSessionGone("MCP session closed")

            async def go():
                return await session.call_tool(name, arguments or {})
            return flatten_result(self._run(go, timeout))

        try:
            return self._with_reconnect(once, idempotent=False)
        except concurrent.futures.TimeoutError:
            return False, f"tool {name} timed out after {timeout:.0f}s"
        except Exception as err:
            cause = _root_cause(err)
            return False, f"MCP error calling {name}: {type(cause).__name__}: {cause}"

    def _with_reconnect(self, fn, idempotent: bool):
        """Run fn; on a broken pipe or closed session, restart the server
        once and retry. A non-idempotent call (a tool call) is retried only
        when the request provably never left ZENIT, so Studio never runs the
        same Luau twice; otherwise the dead session is dropped and the next
        call reconnects."""
        try:
            return fn()
        except (concurrent.futures.TimeoutError, MCPConnectError):
            raise
        except Exception as err:
            cause = _root_cause(err)
            lost = _connection_lost(cause) or not self.connected
            if not lost:
                raise
            self._drop_session()
            if not (idempotent or _send_failed(cause)):
                raise
        return fn()

    def _drop_session(self) -> None:
        with self._lock:
            try:
                self._submit(self._shutdown()).result(10)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
