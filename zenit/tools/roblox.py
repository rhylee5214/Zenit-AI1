"""Roblox Studio tools for ZENIT's agent mode.

The Studio MCP proxy exposes ~40 tools with long descriptions. A 4B model
does better with a short, curated list, so ZENIT shows it eleven tools with
one-line descriptions and small schemas, and maps each call onto the real
MCP tool:

  * studio_id is never shown to the model; the server injects the studio
    that was active when the request started.
  * datamodel_type defaults to "Edit" ("Server" while a playtest that ZENIT
    started is running) when the model leaves it out.
  * light coercion for common slips (numbers sent as strings, code wrapped in
    ``` fences, edits sent as a JSON string, ...).

RobloxService keeps the bridge, the studio list and the active studio.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Callable

from .mcp_bridge import mcp_available

DATAMODEL = {"type": "string", "enum": ["Edit", "Client", "Server"],
             "description": "Edit (default), or Client/Server while playtesting"}

# name -> (description, parameters, required, needs_datamodel)
_CURATED: list[tuple[str, str, dict, list[str], bool]] = [
    ("get_studio_state", "Get Studio's mode (edit or playtest) and which datamodels are available.",
     {}, [], False),
    ("search_game_tree", "List instances in the game tree as JSON. Use before editing to see what exists.",
     {"path": {"type": "string", "description": "start path (default Workspace), e.g. ServerScriptService"},
      "instance_type": {"type": "string", "description": "class filter, e.g. BasePart, Model, Script"},
      "keywords": {"type": "string", "description": "name keywords, comma separated"},
      "max_depth": {"type": "integer", "description": "default 2, max 10"},
      "head_limit": {"type": "integer", "description": "max results, default 60"},
      "datamodel_type": DATAMODEL}, [], True),
    ("inspect_instance", "Show one instance's properties, attributes and children.",
     {"path": {"type": "string", "description": "dot path, e.g. Workspace.House.Door"}}, ["path"], False),
    ("execute_luau", "Run Luau code in Studio and return its output. Use print() to report results.",
     {"code": {"type": "string", "description": "Luau source"}, "datamodel_type": DATAMODEL},
     ["code"], True),
    ("get_console_output", "Read Studio's Output window (prints, warnings, errors).", {}, [], False),
    ("script_read", "Read a script's source.",
     {"target_file": {"type": "string", "description": "dot path, e.g. game.ServerScriptService.Main"},
      "start_line_one_indexed": {"type": "integer"},
      "end_line_one_indexed_inclusive": {"type": "integer"}}, ["target_file"], False),
    ("script_grep", "Search all scripts for a string or Luau pattern.",
     {"query": {"type": "string"}}, ["query"], False),
    ("multi_edit", "Edit a script by exact string replacement, or create it (old_string \"\" + className).",
     {"file_path": {"type": "string", "description": "dot path, e.g. game.ServerScriptService.Main"},
      "edits": {"type": "array", "items": {"type": "object", "properties": {
          "old_string": {"type": "string"}, "new_string": {"type": "string"}},
          "required": ["old_string", "new_string"]}},
      "className": {"type": "string", "description": "Script, LocalScript or ModuleScript (new scripts only)"}},
     ["file_path", "edits"], True),
    ("search_asset", "Search the Creator Store and inventory for models, meshes, audio or images.",
     {"query": {"type": "string"},
      "assetType": {"type": "string", "enum": ["Model", "Audio", "Mesh", "MeshPart", "Image", "Decal"]},
      "maxResults": {"type": "integer", "description": "1-20, default 5"}}, ["query"], False),
    ("insert_asset", "Insert an asset by id into the game.",
     {"assetId": {"type": "string"}, "assetName": {"type": "string"},
      "parentPath": {"type": "string", "description": "e.g. game.Workspace (default)"}}, ["assetId"], False),
    ("start_stop_play", "Start (true) or stop (false) a playtest.",
     {"is_start": {"type": "boolean"}}, ["is_start"], False),
]

CURATED_TOOLS = {name: {"description": d, "parameters": p, "required": r, "datamodel": dm}
                 for name, d, p, r, dm in _CURATED}
TOOL_NAMES = [name for name, *_ in _CURATED]

# Real MCP schemas (properties only) used to drop keys the server would reject.
_MCP_PROPS = {
    "get_studio_state": set(),
    "search_game_tree": {"path", "instance_type", "keywords", "max_depth", "head_limit", "datamodel_type"},
    "inspect_instance": {"path"},
    "execute_luau": {"code", "datamodel_type"},
    "get_console_output": set(),
    "script_read": {"target_file", "should_read_entire_file", "start_line_one_indexed",
                    "end_line_one_indexed_inclusive"},
    "script_grep": {"query"},
    "multi_edit": {"file_path", "edits", "className", "datamodel_type"},
    "search_asset": {"query", "assetType", "maxResults", "scope", "priceFilter", "tags", "facets",
                     "verifiedCreatorsOnly"},
    "insert_asset": {"assetId", "assetName", "assetType", "parentPath"},
    "start_stop_play": {"is_start"},
}

ROBLOX_SYSTEM_PROMPT = """You are ZENIT, a Roblox Studio building assistant. You are connected to the user's open Roblox Studio through tools. Call tools to look at and change the game; never just print code for the user to paste.

How to work:
1. Look first: search_game_tree or inspect_instance shows what already exists.
2. execute_luau changes the place right now (build, move, delete, set properties). It runs once, in Edit mode; event connections made there never fire in the game.
3. Behaviour that must run while the game plays (doors, kill bricks, leaderstats, spawners) goes in a Script: multi_edit with file_path "game.ServerScriptService.<Name>", className "Script", edits [{"old_string": "", "new_string": "<code>"}]. Client-only code goes in a LocalScript in game.StarterPlayer.StarterPlayerScripts. Read an existing script with script_read before editing it.
4. Keep each execute_luau under about 80 lines; build big things in several calls, one section each.

Build script rules:
- First remove the old copy: local old = workspace:FindFirstChild("House"); if old then old:Destroy() end
- Group the build: local m = Instance.new("Model"); m.Name = "House"
- Each part: local p = Instance.new("Part"); p.Name = "Wall"; p.Size = Vector3.new(w, h, d); p.CFrame = CFrame.new(x, y, z); p.Anchored = true; p.Color = Color3.fromRGB(r, g, b); p.Material = Enum.Material.Wood; p.Parent = m (set Parent last)
- The ground top is y = 0, so a part of height h sits on it at y = h/2.
- Rotation is in radians: CFrame.new(x, y, z) * CFrame.Angles(0, math.rad(90), 0)
- Finish with m.PrimaryPart = <base part>; m.Parent = workspace. Move a whole model with m:PivotTo(CFrame.new(x, y, z)).
- Over about 200 parts: set m.Parent = workspace first and call task.wait() after every 200 parts.
- Luau: use local variables, ~= (not !=), .. joins strings, -- starts a comment, blocks close with end, arrays start at 1, task.wait() (never wait()), Instance.new takes one argument.
- End with a print() summary, e.g. print("Built House: 42 parts at 0, 0, 0").

Example execute_luau code:
local old = workspace:FindFirstChild("Hut"); if old then old:Destroy() end
local m = Instance.new("Model"); m.Name = "Hut"
local floor = Instance.new("Part"); floor.Name = "Floor"; floor.Size = Vector3.new(12, 1, 12); floor.CFrame = CFrame.new(0, 0.5, 0); floor.Anchored = true; floor.Material = Enum.Material.WoodPlanks; floor.Parent = m
for i, x in ipairs({-5.5, 5.5}) do
  local w = Instance.new("Part"); w.Name = "Wall" .. i; w.Size = Vector3.new(1, 8, 12); w.CFrame = CFrame.new(x, 5, 0); w.Anchored = true; w.Color = Color3.fromRGB(200, 170, 120); w.Parent = m
end
m.PrimaryPart = floor; m.Parent = workspace
print("Built Hut: 3 parts at 0, 0, 0")

Playtests: build in Edit mode; changes made during a playtest are lost when it stops. Only start a playtest when the user asks, and stop it afterwards.

Honesty: if a tool returns an error, say so, fix the code and try again (the first line removes the broken copy). Never claim something was built unless the tool output confirms it. When done, tell the user in plain words what you built and where it is."""


def tool_specs(names: list[str] | None = None) -> list[dict]:
    """OpenAI/HF style tool list for apply_chat_template(tools=...)."""
    out = []
    for name in names or TOOL_NAMES:
        t = CURATED_TOOLS[name]
        out.append({"type": "function", "function": {
            "name": name, "description": t["description"],
            "parameters": {"type": "object", "properties": t["parameters"], "required": t["required"]}}})
    return out


def _to_int(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v.strip())
    return v


def _to_bool(v):
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
        return v.strip().lower() in ("true", "1", "yes")
    return v


class ArgumentError(ValueError):
    pass


_FENCE = re.compile(r"^\s*```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)\r?\n?[ \t]*```\s*$", re.S)
_CLAMP = {"max_depth": (1, 10), "maxResults": (1, 20), "head_limit": (1, 500),
          "start_line_one_indexed": (1, None), "end_line_one_indexed_inclusive": (1, None)}
READ_WINDOW = 200


def _strip_fences(code):
    if isinstance(code, str):
        m = _FENCE.match(code)
        if m:
            return m.group(1)
    return code


def prepare_arguments(name: str, args: dict | None, studio_id: str | None,
                      default_datamodel: str = "Edit") -> dict:
    """Model arguments -> real MCP arguments (studio_id injected,
    datamodel_type defaulted, unknown keys dropped, light coercion)."""
    if name not in CURATED_TOOLS:
        raise ArgumentError(f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}")
    args = dict(args or {})
    args.pop("studio_id", None)                        # the server decides which studio
    spec = CURATED_TOOLS[name]
    out = {k: v for k, v in args.items() if k in _MCP_PROPS[name]}
    if spec["datamodel"]:
        dm = out.get("datamodel_type")
        if isinstance(dm, str) and dm.strip().lower() in ("edit", "client", "server"):
            out["datamodel_type"] = dm.strip().capitalize()
        else:
            out["datamodel_type"] = default_datamodel
        if name == "multi_edit":
            out["datamodel_type"] = "Edit"             # the real tool only accepts Edit
    for key, (lo, hi) in _CLAMP.items():
        if key in out:
            v = _to_int(out[key])
            if isinstance(v, int) and not isinstance(v, bool):
                v = max(lo, v) if hi is None else max(lo, min(hi, v))
            out[key] = v
    if name == "execute_luau":
        out["code"] = _strip_fences(out.get("code"))
    if name == "search_game_tree":
        if isinstance(out.get("keywords"), (list, tuple)):
            out["keywords"] = ", ".join(str(k) for k in out["keywords"] if k not in (None, ""))
        if not any(out.get(k) for k in ("path", "instance_type", "keywords")):
            out["path"] = "Workspace"                  # the whole DataModel is mostly service noise
            out.setdefault("max_depth", 2)
        out.setdefault("head_limit", 60)
    if name == "script_read":
        start, end = out.get("start_line_one_indexed"), out.get("end_line_one_indexed_inclusive")
        if isinstance(start, int) and not isinstance(end, int):
            out["end_line_one_indexed_inclusive"] = start + READ_WINDOW
        elif isinstance(end, int) and not isinstance(start, int):
            out["start_line_one_indexed"] = 1
        ranged = "start_line_one_indexed" in out or "end_line_one_indexed_inclusive" in out
        out["should_read_entire_file"] = not ranged
    if name == "insert_asset" and "assetId" in out:
        digits = re.sub(r"\D", "", str(out["assetId"]))
        out["assetId"] = digits or str(_to_int(out["assetId"]))
    if name == "start_stop_play" and "is_start" in out:
        out["is_start"] = _to_bool(out["is_start"])
    if name == "multi_edit":
        edits = out.get("edits")
        if isinstance(edits, str) and edits.strip():
            try:
                edits = json.loads(edits, strict=False)
            except ValueError as err:
                raise ArgumentError(f"multi_edit edits must be a JSON array of objects ({err})")
        if isinstance(edits, dict):
            edits = [edits]
        if isinstance(edits, list):
            fixed = []
            for e in edits:
                if isinstance(e, dict):
                    e = dict(e)
                    if "new_string" in e:
                        e.setdefault("old_string", "")
                fixed.append(e)
            edits = fixed
        if edits is not None:
            out["edits"] = edits
    missing = [k for k in spec["required"] if k not in out or out[k] in (None, "")]
    if missing:
        raise ArgumentError(f"{name} needs: {', '.join(missing)}")
    if studio_id:
        out["studio_id"] = studio_id
    return out


_LUAU_ERROR = [
    re.compile(r"^\s*(error|luau error|runtime error|syntax error|script error)\b", re.I),
    re.compile(r"Stack Begin"),
    re.compile(r":\d+: .*(attempt to |is not a valid member|expected |Incomplete statement|unexpected symbol)", re.I),
]


def looks_like_luau_error(text: str) -> bool:
    """True when execute_luau output that the proxy marked successful is
    really a Luau compile or runtime error."""
    return bool(text) and any(p.search(text) for p in _LUAU_ERROR)


def parse_studios(text: str) -> list[dict]:
    """list_roblox_studios output -> [{"id", "name"}]."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    items = data.get("studios", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    out = []
    for s in items:
        if isinstance(s, dict) and s.get("id"):
            out.append({"id": str(s["id"]), "name": str(s.get("name") or s.get("place_name") or s["id"])})
    return out


DEFAULT_CONFIG = {"enabled": True, "command": "cmd.exe",
                  "args": ["/c", "cd /d %LOCALAPPDATA%\\Roblox && .\\mcp.bat"], "env": None,
                  "studio_id": None, "max_steps": 8, "tool_timeout_s": 120, "connect_timeout_s": 30,
                  # Sampling for build mode: code wrapped in JSON needs a low temperature and
                  # no repetition penalty (code legitimately repeats identifiers).
                  "generation": {"temperature": 0.3, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}}


def load_roblox_config(path: str | Path) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    p = Path(path)
    if p.exists():
        cfg.update(json.loads(p.read_text(encoding="utf-8")))
    return cfg


class RobloxService:
    """Owns the MCP bridge (created lazily), the studio list and the active
    studio. Thread-safe; every method may block on the MCP server."""

    def __init__(self, config: dict, bridge_factory: Callable[[dict], object] | None = None) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._factory = bridge_factory
        self.bridge = None
        self.studios: list[dict] = []
        self.studio_id: str | None = self.config.get("studio_id")
        self.error: str | None = None
        self.server_tools: set[str] | None = None
        self.play_mode: dict[str, bool] = {}           # studio_id -> a playtest ZENIT started is running
        self._lock = threading.RLock()

    # --------------------------------------------------------------- state
    @property
    def available(self) -> bool:
        return bool(self.config.get("enabled")) and (self._factory is not None or mcp_available())

    def _make_bridge(self):
        if self._factory is not None:
            return self._factory(self.config)
        from .mcp_bridge import MCPBridge
        return MCPBridge(self.config["command"], self.config.get("args") or [], self.config.get("env"),
                         self.config.get("cwd"), self.config.get("log_file"),
                         connect_timeout=float(self.config.get("connect_timeout_s", 30)))

    def connect(self) -> bool:
        """Open the bridge if needed. Returns connected; sets self.error."""
        with self._lock:
            if not self.config.get("enabled"):
                self.error = "Roblox mode is disabled in configs/roblox.json"
                return False
            if not self.available:
                self.error = "the 'mcp' package is not installed (pip install -e \".[roblox]\")"
                return False
            try:
                if self.bridge is None:
                    self.bridge = self._make_bridge()
                self.bridge.connect(float(self.config.get("connect_timeout_s", 30)))
                if self.server_tools is None:
                    self.server_tools = {t["name"] for t in self.bridge.list_tools()}
                self.error = None
                return True
            except Exception as err:
                self.error = f"could not connect to Roblox Studio MCP: {err}"
                return False

    @property
    def connected(self) -> bool:
        return self.bridge is not None and bool(getattr(self.bridge, "connected", False))

    def refresh_studios(self) -> list[dict]:
        with self._lock:
            ok, text = self.bridge.call_tool("list_roblox_studios", {},
                                             timeout=float(self.config.get("connect_timeout_s", 30)))
            if not ok:
                self.error = f"list_roblox_studios failed: {text[:300]}"
                self.studios = []
            else:
                self.studios = parse_studios(text)
                if not self.studios:
                    self.error = "no Roblox Studio instance is connected (open Studio and enable MCP)"
            ids = [s["id"] for s in self.studios]
            if self.studio_id not in ids:
                preferred = self.config.get("studio_id")
                self.studio_id = preferred if preferred in ids else (ids[0] if ids else None)
            return list(self.studios)

    def exposed_tools(self) -> list[str]:
        if self.server_tools is None:
            return list(TOOL_NAMES)
        return [n for n in TOOL_NAMES if n in self.server_tools]

    def status(self, refresh: bool = True) -> dict:
        with self._lock:
            connected = False
            if self.available and refresh:
                connected = self.connect()
                if connected:
                    self.refresh_studios()
            elif self.available:
                connected = self.connected
            elif not self.config.get("enabled"):
                self.error = "Roblox mode is disabled in configs/roblox.json"
            else:
                self.error = "the 'mcp' package is not installed (pip install -e \".[roblox]\")"
            return {"available": self.available, "connected": bool(connected and self.connected),
                    "studios": list(self.studios), "studio_id": self.studio_id,
                    "tools": self.exposed_tools(), "error": self.error}

    def set_studio(self, studio_id: str) -> bool:
        with self._lock:
            if self.connect():
                self.refresh_studios()
            if studio_id not in [s["id"] for s in self.studios]:
                return False
            self.studio_id = studio_id
            self.error = None
            return True

    # ---------------------------------------------------------------- calls
    def call(self, name: str, arguments: dict | None, studio_id: str | None = None) -> tuple[bool, str]:
        """Execute a curated tool for the model. Never raises. `studio_id`
        pins the studio for a whole agent run (default: the active one)."""
        sid = studio_id or self.studio_id
        try:
            if name not in self.exposed_tools():
                raise ArgumentError(f"unknown tool {name!r}; available: {', '.join(self.exposed_tools())}")
            default_dm = "Server" if sid and self.play_mode.get(sid) else "Edit"
            args = prepare_arguments(name, arguments, sid, default_dm)
        except ArgumentError as err:
            return False, f"Error: {err}"
        if not sid:
            return False, "Error: no Roblox Studio instance is connected"
        if self.bridge is None and not self.connect():
            return False, f"Error: {self.error}"
        try:
            ok, text = self.bridge.call_tool(name, args, timeout=float(self.config.get("tool_timeout_s", 120)))
        except Exception as err:                       # bridges should not raise, but never crash the loop
            return False, f"Error: {type(err).__name__}: {err}"
        if ok and name == "start_stop_play" and isinstance(args.get("is_start"), bool):
            self.play_mode[sid] = args["is_start"]
        if ok and name == "execute_luau" and looks_like_luau_error(text):
            ok = False
        return ok, text

    def close(self) -> None:
        with self._lock:
            if self.bridge is not None:
                try:
                    self.bridge.close()
                finally:
                    self.bridge = None
                    self.server_tools = None
