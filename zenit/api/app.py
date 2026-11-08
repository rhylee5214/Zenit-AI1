"""ZENIT inference API (Starlette).

Every generation request runs on a local ZENIT engine. There is no code
path to any third-party AI provider.

Main endpoint: POST /v1/chat/completions (OpenAI-compatible request and
response shapes, plus ZENIT extras: conversation_id, memory, documents).

Roblox mode: /v1/roblox/status, /v1/roblox/studio and /v1/roblox/chat run a
local tool-calling model against Roblox Studio through its MCP server
(zenit/tools). The MCP bridge starts on first use and stops on shutdown.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .. import __version__
from ..inference.base import GenerationEvent, GenerationParams
from ..inference.registry import ModelPool
from ..knowledge.ingest import MAX_UPLOAD_BYTES, UnsupportedFile
from ..knowledge.store import KnowledgeBase, build_context_block
from ..memory.manager import MemoryManager
from ..persona.identity import addressee_line, load_persona
from ..storage import Database
from ..tools.agent import RobloxAgent
from ..tools.roblox import ROBLOX_SYSTEM_PROMPT, RobloxService, load_roblox_config, tool_specs
from ..train.config import TrainConfig
from ..train.registry import ROADMAP, Registry
from ..train.runlog import read_metrics
from .auth import (Authenticator, RateLimiter, create_key, list_keys, record_usage, revoke_key,
                   tokens_today, usage_report)
from .google import (SESSION_COOKIE, SESSION_TTL_S, GoogleAuthError, GoogleVerifier, create_session,
                     delete_session, load_auth_config, lookup_session, upsert_google_user)
from .system_info import system_report

LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1")
UNSAFE_METHODS = ("POST", "PUT", "PATCH", "DELETE")


# ------------------------------------------------------------------ schemas
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=200_000)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] | None = None        # stateless, OpenAI style
    message: str | None = Field(None, max_length=100_000)   # stateful, ZENIT style
    conversation_id: str | None = None
    regenerate: bool = False
    stream: bool = False
    max_tokens: int | None = Field(None, ge=1, le=32_768)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    top_k: int | None = Field(None, ge=0, le=1000)
    repetition_penalty: float | None = Field(None, ge=0.5, le=2.0)
    seed: int | None = None
    stop: list[str] = Field(default_factory=list, max_length=8)
    profile: str | None = None
    use_memory: bool = True
    use_documents: bool = True
    document_ids: list[str] | None = None


class RobloxChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = None
    model: str | None = None
    max_steps: int | None = Field(None, ge=1, le=16)


class RobloxStudioRequest(BaseModel):
    studio_id: str = Field(min_length=1, max_length=200)


class MemoryIn(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    kind: str = "fact"
    pinned: bool = False


class MemoryPatch(BaseModel):
    content: str | None = Field(None, min_length=1, max_length=2000)
    kind: str | None = None
    pinned: bool | None = None


class ApiError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error",
                 headers: dict | None = None, extra: dict | None = None) -> None:
        super().__init__(message)
        self.status, self.message, self.kind = status, message, kind
        self.headers, self.extra = headers or {}, extra or {}


def sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------- server
class ZenitServer:
    # Tests (or embedders) may replace this: config dict -> bridge object with
    # connect(timeout), close(), list_tools(), call_tool(name, args, timeout)
    # and a `connected` attribute. None means the real stdio MCPBridge.
    roblox_bridge_factory = None

    def __init__(self, root: str | Path = ".", db_path: str | None = None,
                 serve_config: str | None = None, local_mode: bool = False,
                 roblox_bridge=None, roblox_config: dict | None = None) -> None:
        self.root = Path(root).resolve()
        self.db = Database(db_path or self.root / "data" / "zenit.db")
        self.kb = KnowledgeBase(self.db)
        self.mem = MemoryManager(self.db, self.kb)
        self.pool = ModelPool(serve_config or self.root / "configs" / "serve.json", self.root / "models",
                              root=self.root)
        self.persona = load_persona(self.root / "configs" / "persona.json")
        self.auth = Authenticator(self.db)
        self.limiter = RateLimiter()
        self.local_mode = local_mode
        # Sign in with Google switches on when configs/auth.json holds a Web client id.
        self.auth_config = load_auth_config(self.root / "configs" / "auth.json")
        if self.auth_config["error"]:
            print(f"[zenit] Google sign-in disabled: {self.auth_config['error']}")
        cid = self.auth_config["google_client_id"]
        self.google = GoogleVerifier(cid) if cid else None
        # gen_id -> cancel Event; each Event carries .owner (the user_id that started it)
        self.generations: dict[str, threading.Event] = {}
        self.started = time.time()
        factory = getattr(type(self), "roblox_bridge_factory", None)    # class lookup: no method binding
        if roblox_bridge is not None:
            factory = lambda cfg: roblox_bridge          # noqa: E731
        self.roblox = RobloxService(roblox_config if roblox_config is not None
                                    else load_roblox_config(self.root / "configs" / "roblox.json"), factory)

    def shutdown(self) -> None:
        self.roblox.close()

    # ------------------------------------------------------------ plumbing
    def identify(self, request: Request) -> dict:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            row = self.auth.check(header[7:].strip())
            if row:
                return row
            raise ApiError(401, "invalid or revoked API key", "authentication_error")
        if self.google is not None:
            sess = lookup_session(self.db, request.cookies.get(SESSION_COOKIE))
            if sess:
                self.check_csrf(request)
                return self.session_key(request, sess)
        if self.is_local(request):
            return self.auth.local_key()
        raise ApiError(401, "missing API key (Authorization: Bearer zk-...)", "authentication_error")

    def is_local(self, request: Request) -> bool:
        """--local mode and the request comes from this PC (127.0.0.1 or ::1,
        whether the page was opened as http://localhost or http://127.0.0.1)."""
        host = request.client.host if request.client else ""
        return self.local_mode and host in LOOPBACK_HOSTS

    def allowed_origins(self, request: Request) -> set[str]:
        server = request.scope.get("server") or (None, None)
        port = server[1] if len(server) > 1 and server[1] else request.url.port
        hosts = ("localhost", "127.0.0.1", "[::1]")
        out = {f"http://{h}:{port}" for h in hosts}
        if port in (80, None):
            out |= {f"http://{h}" for h in hosts}
        return out

    def check_csrf(self, request: Request) -> None:
        """Cookie-authenticated state changes must come from ZENIT's own page:
        a custom header (cross-site forms cannot set it, and cross-site fetch
        needs a CORS preflight ZENIT does not grant) plus, when the browser
        sends one, an Origin of http://localhost:<port> or http://127.0.0.1:<port>."""
        if request.method not in UNSAFE_METHODS:
            return
        if request.headers.get("x-zenit-client") != "web":
            raise ApiError(403, "missing X-Zenit-Client header", "permission_error")
        origin = request.headers.get("origin")
        if origin is not None and origin not in self.allowed_origins(request):
            raise ApiError(403, "request origin not allowed", "permission_error")

    def session_key(self, request: Request, sess: dict) -> dict:
        """Key dict for a Google session: the user's own key row (rate limit,
        usage) plus who they are. In --local mode a signed-in user on this PC
        keeps the admin rights they would have without signing in."""
        key = dict(sess["key"])
        acc = sess["account"]
        if self.is_local(request):
            key["is_admin"] = 1
        key["person"] = {"name": acc["name"], "given_name": acc["given_name"], "email": acc["email"],
                         "picture": acc["picture"]}
        return key

    @staticmethod
    def person_name(key: dict | None) -> str | None:
        """Name for the system prompt (signed-in users only; never the email)."""
        p = (key or {}).get("person") or {}
        return p.get("given_name") or p.get("name") or None

    def route(self, fn, auth: bool = True, admin: bool = False):
        async def handler(request: Request):
            try:
                key = self.identify(request) if auth else None
                if admin and not key["is_admin"]:
                    raise ApiError(403, "admin key required", "permission_error")
                return await fn(request, key)
            except ApiError as err:
                return JSONResponse({"error": {"message": err.message, "type": err.kind, **err.extra}},
                                    status_code=err.status, headers=err.headers)
            except ValidationError as err:
                return JSONResponse({"error": {"message": "invalid request", "type": "invalid_request_error",
                                               "details": json.loads(err.json())}}, status_code=422)
        return handler

    @staticmethod
    async def body(request: Request) -> dict:
        try:
            data = await request.json()
        except Exception:
            raise ApiError(400, "body must be valid JSON")
        if not isinstance(data, dict):
            raise ApiError(400, "body must be a JSON object")
        return data

    # ------------------------------------------------------------- general
    async def health(self, request, key):
        return JSONResponse({"status": "ok", "version": __version__, "uptime_s": round(time.time() - self.started),
                             "models": len(self.pool.engines), "default_model": self.pool.default})

    async def me(self, request, key):
        person = key.get("person") or {}
        return JSONResponse({"user_id": key["user_id"], "key_prefix": key["prefix"], "is_admin": bool(key["is_admin"]),
                             "rate_per_min": key["rate_per_min"], "local_mode": key["label"] == "local-mode",
                             "signed_in": bool(person), "name": person.get("name"),
                             "given_name": person.get("given_name"), "email": person.get("email"),
                             "picture": person.get("picture"), "google_enabled": self.google is not None})

    # ---------------------------------------------------------------- auth
    async def auth_config_view(self, request, key):
        cid = self.auth_config["google_client_id"] if self.google else None
        return JSONResponse({"google_client_id": cid, "enabled": self.google is not None,
                             "local_access": self.is_local(request)},
                            headers={"Cache-Control": "no-store"})

    async def auth_google(self, request, key):
        if self.google is None:
            raise ApiError(404, "Google sign-in is not set up (add google_client_id to configs/auth.json)")
        self.check_csrf(request)
        host = request.client.host if request.client else "?"
        ok, retry = self.limiter.allow("auth-google:" + host, 30)
        if not ok:
            raise ApiError(429, "too many sign-in attempts", "rate_limit_error", {"Retry-After": str(int(retry) + 1)})
        data = await self.body(request)
        credential = data.get("credential")
        if not isinstance(credential, str) or not credential:
            raise ApiError(400, "credential (a Google ID token) is required")
        try:
            claims = await run_in_threadpool(self.google.verify, credential)
        except GoogleAuthError as err:
            raise ApiError(err.status, err.message,
                           "service_unavailable" if err.status == 503 else "authentication_error")
        account = await run_in_threadpool(upsert_google_user, self.db, claims)
        delete_session(self.db, request.cookies.get(SESSION_COOKIE))     # replace any older session
        token = create_session(self.db, account["user_id"])
        resp = JSONResponse({"signed_in": True, "user_id": account["user_id"], "name": account["name"],
                             "given_name": account["given_name"], "email": account["email"],
                             "picture": account["picture"]})
        resp.set_cookie(SESSION_COOKIE, token, max_age=SESSION_TTL_S, path="/", httponly=True, samesite="lax",
                        secure=request.url.scheme == "https")
        return resp

    async def auth_logout(self, request, key):
        self.check_csrf(request)
        removed = delete_session(self.db, request.cookies.get(SESSION_COOKIE))
        resp = JSONResponse({"signed_out": True, "had_session": removed})
        resp.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax",
                           secure=request.url.scheme == "https")
        return resp

    async def models(self, request, key):
        data = await run_in_threadpool(self.pool.list)
        return JSONResponse({"object": "list", "data": data, "default": self.pool.default,
                             "profiles": {n: p.description for n, p in self.persona.profiles.items()}})

    async def model_detail(self, request, key):
        mid = request.path_params["model_id"]
        try:
            eng = await run_in_threadpool(self.pool.get, mid)
        except KeyError:
            raise ApiError(404, f"unknown model {mid}")
        return JSONResponse(await run_in_threadpool(eng.info))

    # ---------------------------------------------------------------- chat
    def check_limits(self, key: dict) -> None:
        ok, retry = self.limiter.allow(key["id"], key["rate_per_min"])
        if not ok:
            raise ApiError(429, "rate limit reached", "rate_limit_error", {"Retry-After": str(int(retry) + 1)})
        if key["daily_token_limit"] and tokens_today(self.db, key["id"]) >= key["daily_token_limit"]:
            raise ApiError(429, "daily token limit reached", "rate_limit_error")

    async def chat(self, request, key):
        req = ChatRequest(**await self.body(request))
        self.check_limits(key)
        try:
            engine = await run_in_threadpool(self.pool.get, req.model)
        except KeyError as err:
            raise ApiError(404, str(err).strip("'"))
        user_id = key["user_id"]
        prof = self.persona.profile(req.profile)
        base = self.persona.sampling(req.profile, getattr(engine, "generation", None))
        params = GenerationParams(
            max_tokens=min(req.max_tokens or base["max_tokens"], engine.context_window),
            temperature=base["temperature"] if req.temperature is None else req.temperature,
            top_p=base["top_p"] if req.top_p is None else req.top_p,
            top_k=base["top_k"] if req.top_k is None else req.top_k,
            repetition_penalty=base["repetition_penalty"] if req.repetition_penalty is None else req.repetition_penalty,
            seed=req.seed, stop=req.stop)

        conv_id, actions, memories, sources, dropped = None, [], [], [], 0
        if req.message is None and not req.regenerate:
            if not req.messages:
                raise ApiError(400, "provide message (with optional conversation_id) or messages")
            prompt = [m.model_dump() for m in req.messages]
            if prompt[0]["role"] != "system":
                prompt.insert(0, {"role": "system", "content": self.persona.system_for(
                    req.profile, getattr(engine, "card", {}), short=engine.short_system_prompt,
                    user_name=self.person_name(key))})
            last_user = next((m["content"] for m in reversed(prompt) if m["role"] == "user"), "")
            if req.use_documents and req.document_ids and last_user:
                hits = self.kb.search(user_id, last_user, k=6, doc_ids=req.document_ids)
                block, sources = build_context_block(hits, engine.context_window // 3, engine.count_tokens)
                if block:
                    prompt.insert(1, {"role": "context", "content": block})
        else:
            if req.conversation_id:
                conv = self.mem.get_conversation(user_id, req.conversation_id)
                if not conv:
                    raise ApiError(404, "conversation not found")
            else:
                if req.regenerate:
                    raise ApiError(400, "regenerate needs conversation_id")
                conv = self.mem.create_conversation(user_id, (req.message or "New chat")[:60], engine.id)
            conv_id = conv["id"]
            history = self.mem.messages(conv_id)
            if req.regenerate:
                self.mem.pop_last_assistant(conv_id)
                history = self.mem.messages(conv_id)
                if not history or history[-1]["role"] != "user":
                    raise ApiError(400, "nothing to regenerate")
                new_message, history = history[-1]["content"], history[:-1]
            else:
                new_message = req.message
                if not history and conv["title"] in ("New chat", None, ""):
                    self.mem.rename_conversation(user_id, conv_id, new_message[:60])
                self.mem.add_message(conv_id, "user", new_message)
                if req.use_memory:
                    actions = self.mem.process_user_message(user_id, new_message, conv_id)
            built = await run_in_threadpool(
                self.mem.build_messages, user_id, engine, self.persona, req.profile, history, new_message,
                params.max_tokens, req.use_memory, req.use_documents, req.document_ids, conv.get("summary"),
                self.person_name(key))
            prompt, sources, memories, dropped = built["messages"], built["sources"], built["memories"], built["dropped_turns"]

        gen_id = "gen_" + uuid.uuid4().hex[:16]
        cancel = threading.Event()
        cancel.owner = user_id
        self.generations[gen_id] = cancel
        zenit_meta = {"generation_id": gen_id, "conversation_id": conv_id, "model": engine.id,
                      "sources": sources, "memories": memories, "memory_actions": actions,
                      "dropped_turns": dropped, "profile": prof.name}
        created = int(time.time())

        def finish(text: str, ev: GenerationEvent) -> None:
            if conv_id:
                self.mem.add_message(conv_id, "assistant", text, engine.id,
                                     {"sources": sources, "finish_reason": ev.finish_reason,
                                      "memories": memories, "generation_id": gen_id},
                                     ev.prompt_tokens, ev.completion_tokens)
            record_usage(self.db, key["id"], engine.id, ev.prompt_tokens, ev.completion_tokens)
            self.generations.pop(gen_id, None)

        if not req.stream:
            def run():
                parts, last = [], GenerationEvent("done", finish_reason="stop")
                for ev in engine.stream(prompt, params, cancel):
                    if ev.type == "token":
                        parts.append(ev.text)
                    else:
                        last = ev
                return "".join(parts), last
            text, ev = await run_in_threadpool(run)
            finish(text, ev)
            return JSONResponse({
                "id": gen_id, "object": "chat.completion", "created": created, "model": engine.id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": ev.finish_reason}],
                "usage": {"prompt_tokens": ev.prompt_tokens, "completion_tokens": ev.completion_tokens,
                          "total_tokens": ev.prompt_tokens + ev.completion_tokens},
                "zenit": zenit_meta})

        async def events():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def worker():
                try:
                    for ev in engine.stream(prompt, params, cancel):
                        loop.call_soon_threadsafe(queue.put_nowait, ev)
                except Exception as err:            # surface engine errors to the client
                    loop.call_soon_threadsafe(queue.put_nowait, err)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            threading.Thread(target=worker, daemon=True).start()
            base = {"id": gen_id, "object": "chat.completion.chunk", "created": created, "model": engine.id}
            parts: list[str] = []
            done: GenerationEvent | None = None
            try:
                yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                           "zenit": zenit_meta})
                while True:
                    ev = await queue.get()
                    if ev is None:
                        break
                    if isinstance(ev, Exception):
                        yield sse({"error": {"message": f"engine error: {ev}", "type": "engine_error"}})
                        break
                    if ev.type == "token":
                        parts.append(ev.text)
                        yield sse({**base, "choices": [{"index": 0, "delta": {"content": ev.text}, "finish_reason": None}]})
                    else:
                        done = ev
                if done is not None:
                    yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": done.finish_reason}],
                               "usage": {"prompt_tokens": done.prompt_tokens, "completion_tokens": done.completion_tokens,
                                         "total_tokens": done.prompt_tokens + done.completion_tokens}})
                yield "data: [DONE]\n\n"
            finally:
                cancel.set()
                if done is None:
                    done = GenerationEvent("done", finish_reason="cancelled", completion_tokens=len(parts))
                finish("".join(parts), done)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def cancel_generation(self, request, key):
        ev = self.generations.get(request.path_params["gen_id"])
        if ev is not None and getattr(ev, "owner", key["user_id"]) != key["user_id"]:
            ev = None                                   # only your own generations
        if ev:
            ev.set()
        return JSONResponse({"cancelled": bool(ev)})

    # -------------------------------------------------------------- roblox
    def roblox_payload(self, data: dict) -> dict:
        default = self.pool.peek(None)
        return {**data, "model_supports_tools": bool(getattr(default, "supports_tools", False))}

    async def roblox_status(self, request, key):
        data = await run_in_threadpool(self.roblox.status)
        return JSONResponse(self.roblox_payload(data))

    def roblox_unavailable(self) -> ApiError:
        msg = self.roblox.status(refresh=False)["error"] or "Roblox mode is unavailable"
        return ApiError(503, msg, "service_unavailable")

    async def roblox_studio(self, request, key):
        req = RobloxStudioRequest(**await self.body(request))
        if not self.roblox.available:
            raise self.roblox_unavailable()
        if not await run_in_threadpool(self.roblox.set_studio, req.studio_id):
            if not self.roblox.connected:
                raise ApiError(503, self.roblox.error or "Roblox Studio MCP is not connected", "service_unavailable")
            raise ApiError(404, f"unknown studio {req.studio_id}")
        return JSONResponse(self.roblox_payload(self.roblox.status(refresh=False)))

    def roblox_prompt(self, engine, history: list[dict], new_message: str,
                      user_name: str | None = None) -> list[dict]:
        """Roblox system prompt + as many recent turns as fit in half the
        context window. Earlier Roblox replies carry a short note of the
        tools they used, with the first output line of each successful build
        or script edit (its print summary), so the model knows what exists."""
        system = ROBLOX_SYSTEM_PROMPT
        card = getattr(engine, "card", {}) or {}
        if card.get("base_model"):
            system += ("\n\nModel card facts: ZENIT " + str(card.get("version", "")) +
                       " runs locally on the open-weight model " + str(card["base_model"]) + ".")
        who = addressee_line(user_name)
        if who:
            system += "\n\n" + who
        budget = (max(512, engine.context_window // 2) - engine.count_tokens(system)
                  - engine.count_tokens(new_message))
        kept: list[dict] = []
        for h in reversed(history):
            if h["role"] not in ("user", "assistant"):
                continue
            content = h["content"] or ""
            calls = (h.get("meta") or {}).get("tool_calls") or []
            if h["role"] == "assistant" and calls:
                notes = []
                for c in calls[:12]:
                    note = "%s %s" % (c.get("name"), "ok" if c.get("ok") else "error")
                    if c.get("ok") and c.get("name") in ("execute_luau", "multi_edit"):
                        first = next((ln.strip() for ln in str(c.get("output") or "").splitlines()
                                      if ln.strip()), "")
                        if first:
                            note += ": " + first[:160]
                    notes.append(note)
                content = (content + "\n(Earlier tool results: " + "; ".join(notes) + ")").strip()
            cost = engine.count_tokens(content) + 4
            if cost > budget:
                break
            budget -= cost
            kept.insert(0, {"role": h["role"], "content": content})
        return [{"role": "system", "content": system}] + kept + [{"role": "user", "content": new_message}]

    async def roblox_chat(self, request, key):
        req = RobloxChatRequest(**await self.body(request))
        self.check_limits(key)
        candidate = self.pool.peek(req.model)          # no weight loading before the cheap checks
        if candidate is None:
            raise ApiError(404, f"model not available: {req.model or self.pool.default}")
        if not getattr(candidate, "supports_tools", False):
            raise ApiError(400, f"model {candidate.id} does not support tool calls; choose a model with "
                                "supports_tools in configs/serve.json")
        if not self.roblox.available:
            raise self.roblox_unavailable()
        if not await run_in_threadpool(self.roblox.connect):
            raise ApiError(503, self.roblox.error or "cannot connect to Roblox Studio MCP", "service_unavailable")
        if not self.roblox.studio_id or not self.roblox.studios:
            await run_in_threadpool(self.roblox.refresh_studios)
        if not self.roblox.studio_id:
            raise ApiError(503, self.roblox.error or "no Roblox Studio instance is connected", "service_unavailable")
        try:
            engine = await run_in_threadpool(self.pool.get, candidate.id)
        except KeyError as err:
            raise ApiError(404, str(err).strip("'"))

        user_id = key["user_id"]
        if req.conversation_id:
            conv = self.mem.get_conversation(user_id, req.conversation_id)
            if not conv:
                raise ApiError(404, "conversation not found")
        else:
            conv = self.mem.create_conversation(user_id, req.message[:60], engine.id)
        conv_id = conv["id"]
        history = self.mem.messages(conv_id)
        if not history and conv["title"] in ("New chat", None, ""):
            self.mem.rename_conversation(user_id, conv_id, req.message[:60])
        self.mem.add_message(conv_id, "user", req.message)
        prompt = await run_in_threadpool(self.roblox_prompt, engine, history, req.message,
                                         self.person_name(key))

        # Build mode samples with configs/roblox.json "generation" (low temperature, no
        # repetition penalty); keys it leaves out fall back to the model's chat defaults.
        base = dict(self.persona.sampling(None, getattr(engine, "generation", None)))
        for k, v in (self.roblox.config.get("generation") or {}).items():
            if k in ("temperature", "top_p", "top_k", "repetition_penalty") and v is not None:
                base[k] = v
        params = GenerationParams(max_tokens=256,
                                  temperature=float(base["temperature"]), top_p=float(base["top_p"]),
                                  top_k=int(base["top_k"]), repetition_penalty=float(base["repetition_penalty"]))
        max_steps = req.max_steps or int(self.roblox.config.get("max_steps", 8))
        max_steps = max(1, min(16, max_steps))
        studio_id = self.roblox.studio_id            # pinned for the whole run
        agent = RobloxAgent(lambda name, args: self.roblox.call(name, args, studio_id),
                            tool_specs(self.roblox.exposed_tools()))
        was_playing = bool(self.roblox.play_mode.get(studio_id))

        gen_id = "gen_" + uuid.uuid4().hex[:16]
        cancel = threading.Event()
        cancel.owner = user_id
        self.generations[gen_id] = cancel
        saved = {"done": False}

        def finish(finish_reason: str, steps: int, error: str | None = None) -> None:
            if saved["done"]:
                return
            saved["done"] = True
            calls = [{"name": c["name"], "arguments": c["arguments"], "ok": c["ok"],
                      "output": (c["output"] or "")[:500]} for c in list(agent.calls)]
            text = re.sub(r"\n{3,}", "\n\n", "".join(agent.prose)).strip()
            meta = {"roblox": True, "tool_calls": calls, "generation_id": gen_id,
                    "finish_reason": finish_reason, "steps": steps, "studio_id": studio_id}
            if error:
                meta["error"] = error[:500]
            self.mem.add_message(conv_id, "assistant", text, engine.id, meta,
                                 agent.prompt_tokens, agent.completion_tokens)
            record_usage(self.db, key["id"], engine.id, agent.prompt_tokens, agent.completion_tokens)
            self.generations.pop(gen_id, None)

        async def events():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def post(item) -> None:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except RuntimeError:                  # event loop closed (server shutting down)
                    pass

            def worker():
                # The worker, not the SSE generator, saves the turn: when the client
                # disconnects (the UI's Stop), a tool call already running in Studio
                # still finishes and must be recorded.
                last: dict = {"done": None, "error": None}
                try:
                    for ev in agent.run(engine, prompt, params, cancel, max_steps):
                        if ev.get("type") == "done":
                            last["done"] = ev
                        elif ev.get("type") == "error":
                            last["error"] = ev.get("message")
                        post(ev)
                except Exception as err:
                    last["done"] = {"finish_reason": "error", "steps": 0}
                    last["error"] = f"agent error: {type(err).__name__}: {err}"
                    post(err)
                finally:
                    try:
                        if not was_playing and self.roblox.play_mode.get(studio_id):
                            self.roblox.call("start_stop_play", {"is_start": False}, studio_id)
                        done_ev = last["done"] or {"finish_reason": "cancelled" if cancel.is_set() else "error",
                                                   "steps": 0}
                        finish(done_ev["finish_reason"], done_ev.get("steps", 0), last["error"])
                    finally:
                        post(None)

            threading.Thread(target=worker, daemon=True).start()
            try:
                yield sse({"type": "meta", "conversation_id": conv_id, "generation_id": gen_id,
                           "model": engine.id, "studio_id": studio_id})
                while True:
                    ev = await queue.get()
                    if ev is None:                   # posted after the worker saved the turn
                        break
                    if isinstance(ev, Exception):
                        yield sse({"type": "error", "message": f"agent error: {type(ev).__name__}: {ev}"})
                        yield sse({"type": "done", "finish_reason": "error", "steps": 0})
                        continue
                    yield sse(ev)
                yield "data: [DONE]\n\n"
            finally:
                cancel.set()                         # the worker saves the turn once it stops

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------- conversations
    async def list_conversations(self, request, key):
        return JSONResponse({"data": self.mem.list_conversations(key["user_id"])})

    async def create_conversation(self, request, key):
        data = await self.body(request) if await request.body() else {}
        conv = self.mem.create_conversation(key["user_id"], str(data.get("title", "New chat"))[:200], data.get("model", ""))
        return JSONResponse(conv, status_code=201)

    async def get_conversation(self, request, key):
        conv = self.mem.get_conversation(key["user_id"], request.path_params["cid"])
        if not conv:
            raise ApiError(404, "conversation not found")
        return JSONResponse({**conv, "messages": self.mem.messages(conv["id"])})

    async def patch_conversation(self, request, key):
        data = await self.body(request)
        cid = request.path_params["cid"]
        if not self.mem.get_conversation(key["user_id"], cid):
            raise ApiError(404, "conversation not found")
        if "title" in data:
            self.mem.rename_conversation(key["user_id"], cid, str(data["title"]))
        return JSONResponse(self.mem.get_conversation(key["user_id"], cid))

    async def delete_conversation(self, request, key):
        if not self.mem.delete_conversation(key["user_id"], request.path_params["cid"]):
            raise ApiError(404, "conversation not found")
        return JSONResponse({"deleted": True})

    # ------------------------------------------------------------ memories
    async def list_memories(self, request, key):
        uid = key["user_id"]
        return JSONResponse({"enabled": self.mem.enabled(uid), "data": self.mem.list_memories(uid)})

    async def add_memory(self, request, key):
        m = MemoryIn(**await self.body(request))
        return JSONResponse(self.mem.add_memory(key["user_id"], m.content, m.kind, m.pinned), status_code=201)

    async def patch_memory(self, request, key):
        p = MemoryPatch(**await self.body(request))
        out = self.mem.update_memory(key["user_id"], request.path_params["mid"], p.content, p.pinned, p.kind)
        if not out:
            raise ApiError(404, "memory not found")
        return JSONResponse(out)

    async def delete_memory(self, request, key):
        if not self.mem.delete_memory(key["user_id"], request.path_params["mid"]):
            raise ApiError(404, "memory not found")
        return JSONResponse({"deleted": True})

    async def delete_all_memories(self, request, key):
        return JSONResponse({"deleted": self.mem.delete_all_memories(key["user_id"])})

    async def export(self, request, key):
        data = await run_in_threadpool(self.mem.export, key["user_id"])
        return JSONResponse(data, headers={"Content-Disposition": "attachment; filename=zenit-export.json"})

    async def get_settings(self, request, key):
        uid = key["user_id"]
        return JSONResponse({"memory_enabled": self.mem.enabled(uid),
                             "default_profile": self.db.get_setting(uid, "default_profile", self.persona.default_profile)})

    async def put_settings(self, request, key):
        data = await self.body(request)
        uid = key["user_id"]
        if "memory_enabled" in data:
            self.db.set_setting(uid, "memory_enabled", "true" if data["memory_enabled"] else "false")
        if "default_profile" in data:
            if data["default_profile"] not in self.persona.profiles:
                raise ApiError(400, "unknown profile")
            self.db.set_setting(uid, "default_profile", data["default_profile"])
        return await self.get_settings(request, key)

    # ----------------------------------------------------------- documents
    async def upload_document(self, request, key):
        ok, retry = self.limiter.allow(key["id"] + ":upload", max(5, key["rate_per_min"] // 2))
        if not ok:
            raise ApiError(429, "upload rate limit reached", "rate_limit_error", {"Retry-After": str(int(retry) + 1)})
        length = int(request.headers.get("content-length") or 0)
        if length > MAX_UPLOAD_BYTES:
            raise ApiError(413, "file larger than 25 MB")
        ctype = request.headers.get("content-type", "")
        filename = request.query_params.get("filename") or request.headers.get("x-filename", "")
        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            f = form.get("file")
            if f is None:
                raise ApiError(400, "multipart upload needs a 'file' field")
            data, filename = await f.read(), f.filename
        else:
            data = await request.body()
        if not filename:
            raise ApiError(400, "filename required (?filename=report.pdf)")
        if not data:
            raise ApiError(400, "empty file")
        try:
            doc = await run_in_threadpool(self.kb.add_document, key["user_id"], Path(filename).name, data, ctype)
        except (UnsupportedFile, ValueError) as err:
            raise ApiError(415, str(err))
        return JSONResponse(doc, status_code=201)

    async def list_documents(self, request, key):
        return JSONResponse({"data": self.kb.list_documents(key["user_id"])})

    async def delete_document(self, request, key):
        if not self.kb.delete_document(key["user_id"], request.path_params["doc_id"]):
            raise ApiError(404, "document not found")
        return JSONResponse({"deleted": True})

    async def search_documents(self, request, key):
        data = await self.body(request)
        q = str(data.get("query", "")).strip()
        if not q:
            raise ApiError(400, "query required")
        hits = await run_in_threadpool(self.kb.search, key["user_id"], q, int(data.get("k", 5)), data.get("document_ids"))
        return JSONResponse({"data": hits})

    # --------------------------------------------------------- usage, keys
    async def usage(self, request, key):
        scope = request.query_params.get("scope", "user")
        if scope == "key":
            return JSONResponse(usage_report(self.db, key_id=key["id"]))
        return JSONResponse(usage_report(self.db, user_id=key["user_id"]))

    async def create_key(self, request, key):
        data = await self.body(request)
        new = create_key(self.db, str(data.get("user_name", "user")), str(data.get("label", "")),
                         bool(data.get("is_admin", False)), int(data.get("rate_per_min", 30)),
                         int(data.get("daily_token_limit", 0)), data.get("user_id"))
        return JSONResponse(new, status_code=201)

    async def list_keys(self, request, key):
        return JSONResponse({"data": list_keys(self.db)})

    async def revoke_key(self, request, key):
        return JSONResponse({"revoked": revoke_key(self.db, request.path_params["key_id"])})

    # ------------------------------------------------------------ training
    def _runs_dir(self) -> Path:
        return self.root / "runs"

    async def list_runs(self, request, key):
        out = []
        for f in sorted(self._runs_dir().glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            out.append({k: s.get(k) for k in ("run_id", "run_name", "version", "status", "step", "max_steps",
                                              "last_loss", "best_val_loss", "last_val_loss", "parameters",
                                              "backend", "device", "created", "updated", "elapsed_s")})
        return JSONResponse({"data": out})

    async def get_run(self, request, key):
        rid = request.path_params["run_id"]
        d = self._runs_dir() / rid
        if not (d / "run.json").exists() or "/" in rid or ".." in rid:
            raise ApiError(404, "run not found")
        state = json.loads((d / "run.json").read_text(encoding="utf-8"))
        metrics = read_metrics(d)
        train = [m for m in metrics if "loss" in m]
        evals = [m for m in metrics if "val_loss" in m]
        stride = max(1, len(train) // 1500)
        return JSONResponse({"run": state, "train": train[::stride] + train[-1:] if train else [],
                             "eval": evals, "latest": train[-1] if train else None})

    async def stop_run(self, request, key):
        d = self._runs_dir() / request.path_params["run_id"]
        if not (d / "run.json").exists():
            raise ApiError(404, "run not found")
        (d / "STOP").write_text("stop requested via API\n", encoding="utf-8")
        return JSONResponse({"stop_requested": True})

    def _train_config(self, data: dict) -> TrainConfig:
        cfg = data.get("config", data)
        try:
            return TrainConfig(**{k: v for k, v in cfg.items() if k in TrainConfig.__dataclass_fields__})
        except (TypeError, ValueError) as err:
            raise ApiError(400, f"bad training config: {err}")

    async def estimate_run(self, request, key):
        from ..tokenizer.bpe import ZenitTokenizer
        from ..train.estimate import estimate, needs_confirmation
        tc = self._train_config(await self.body(request))
        tok_path = self.root / tc.tokenizer
        vocab = ZenitTokenizer.load(tok_path).vocab_size if tok_path.exists() else None
        est = await run_in_threadpool(estimate, tc, vocab)
        return JSONResponse({"estimate": est, "needs_confirmation": needs_confirmation(est)})

    async def start_run(self, request, key):
        from ..tokenizer.bpe import ZenitTokenizer
        from ..train.estimate import estimate, needs_confirmation
        data = await self.body(request)
        tc = self._train_config(data)
        if not (self.root / tc.data_dir / "meta.json").exists():
            raise ApiError(400, f"dataset not built: {tc.data_dir}")
        vocab = ZenitTokenizer.load(self.root / tc.tokenizer).vocab_size
        est = await run_in_threadpool(estimate, tc, vocab)
        reasons = needs_confirmation(est)
        if reasons and not data.get("confirm"):
            raise ApiError(409, "confirmation required", "confirmation_required",
                           extra={"reasons": reasons, "estimate": est})
        cfg_dir = self._runs_dir() / "_configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cfg_path = cfg_dir / f"{tc.run_name}-{stamp}.json"
        cfg_path.write_text(json.dumps(tc.to_dict(), indent=2), encoding="utf-8")
        log = open(cfg_dir / f"{tc.run_name}-{stamp}.log", "w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-m", "zenit.train", "--config", str(cfg_path), "--yes"],
                                cwd=self.root, stdout=log, stderr=subprocess.STDOUT)
        return JSONResponse({"started": True, "pid": proc.pid, "config": str(cfg_path.relative_to(self.root)),
                             "estimate": est}, status_code=202)

    # ---------------------------------------------------- registry, system
    async def registry(self, request, key):
        return JSONResponse({"models": Registry(self.root / "models").list(), "roadmap": ROADMAP})

    async def evals(self, request, key):
        out = []
        for f in sorted((self.root / "evals" / "results").glob("*.json"), reverse=True):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
                out.append({"file": f.name, "model": r.get("model"), "created": r.get("created"),
                            "suite_hash": r.get("suite_hash"), "summary": r.get("summary")})
            except json.JSONDecodeError:
                continue
        return JSONResponse({"data": out})

    async def system(self, request, key):
        return JSONResponse(await run_in_threadpool(system_report))

    # --------------------------------------------------------------- pages
    async def page(self, name: str):
        f = self.root / "web" / name
        if not f.exists():
            return JSONResponse({"error": {"message": f"{name} missing"}}, status_code=404)
        return FileResponse(f, headers={"Cache-Control": "no-cache"})


def create_app(root: str | Path = ".", db_path: str | None = None, serve_config: str | None = None,
               local_mode: bool = False, roblox_bridge=None, roblox_config: dict | None = None) -> Starlette:
    s = ZenitServer(root, db_path, serve_config, local_mode, roblox_bridge, roblox_config)
    r = s.route

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await run_in_threadpool(s.shutdown)         # stops the Roblox MCP server process

    async def index(request):
        return await s.page("index.html")

    async def dashboard(request):
        return await s.page("dashboard.html")

    routes = [
        Route("/", index),
        Route("/dashboard", dashboard),
        Route("/v1/health", r(s.health, auth=False)),
        Route("/v1/me", r(s.me)),
        Route("/v1/auth/config", r(s.auth_config_view, auth=False), methods=["GET"]),
        Route("/v1/auth/google", r(s.auth_google, auth=False), methods=["POST"]),
        Route("/v1/auth/logout", r(s.auth_logout, auth=False), methods=["POST"]),
        Route("/v1/models", r(s.models)),
        Route("/v1/models/{model_id}", r(s.model_detail)),
        Route("/v1/chat/completions", r(s.chat), methods=["POST"]),
        Route("/v1/generations/{gen_id}/cancel", r(s.cancel_generation), methods=["POST"]),
        Route("/v1/roblox/status", r(s.roblox_status), methods=["GET"]),
        Route("/v1/roblox/studio", r(s.roblox_studio), methods=["POST"]),
        Route("/v1/roblox/chat", r(s.roblox_chat), methods=["POST"]),
        Route("/v1/conversations", r(s.list_conversations), methods=["GET"]),
        Route("/v1/conversations", r(s.create_conversation), methods=["POST"]),
        Route("/v1/conversations/{cid}", r(s.get_conversation), methods=["GET"]),
        Route("/v1/conversations/{cid}", r(s.patch_conversation), methods=["PATCH"]),
        Route("/v1/conversations/{cid}", r(s.delete_conversation), methods=["DELETE"]),
        Route("/v1/memories", r(s.list_memories), methods=["GET"]),
        Route("/v1/memories", r(s.add_memory), methods=["POST"]),
        Route("/v1/memories", r(s.delete_all_memories), methods=["DELETE"]),
        Route("/v1/memories/{mid}", r(s.patch_memory), methods=["PATCH"]),
        Route("/v1/memories/{mid}", r(s.delete_memory), methods=["DELETE"]),
        Route("/v1/export", r(s.export)),
        Route("/v1/settings", r(s.get_settings), methods=["GET"]),
        Route("/v1/settings", r(s.put_settings), methods=["PUT"]),
        Route("/v1/documents", r(s.list_documents), methods=["GET"]),
        Route("/v1/documents", r(s.upload_document), methods=["POST"]),
        Route("/v1/documents/search", r(s.search_documents), methods=["POST"]),
        Route("/v1/documents/{doc_id}", r(s.delete_document), methods=["DELETE"]),
        Route("/v1/usage", r(s.usage)),
        Route("/v1/keys", r(s.list_keys, admin=True), methods=["GET"]),
        Route("/v1/keys", r(s.create_key, admin=True), methods=["POST"]),
        Route("/v1/keys/{key_id}", r(s.revoke_key, admin=True), methods=["DELETE"]),
        Route("/v1/training/runs", r(s.list_runs), methods=["GET"]),
        Route("/v1/training/runs", r(s.start_run, admin=True), methods=["POST"]),
        Route("/v1/training/estimate", r(s.estimate_run, admin=True), methods=["POST"]),
        Route("/v1/training/runs/{run_id}", r(s.get_run)),
        Route("/v1/training/runs/{run_id}/stop", r(s.stop_run, admin=True), methods=["POST"]),
        Route("/v1/registry", r(s.registry)),
        Route("/v1/evals", r(s.evals)),
        Route("/v1/system", r(s.system)),
        Mount("/static", StaticFiles(directory=str(s.root / "web" / "static"), check_dir=False), name="static"),
    ]
    middleware = []
    origins = [o for o in os.environ.get("ZENIT_CORS_ORIGINS", "").split(",") if o]
    if origins:
        middleware.append(Middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"],
                                     allow_headers=["*"]))
    app = Starlette(routes=routes, middleware=middleware, lifespan=lifespan)
    app.state.zenit = s
    return app
