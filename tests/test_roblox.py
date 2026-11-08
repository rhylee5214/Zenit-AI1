"""Roblox mode: tool-call parsing, the agent loop, argument mapping, the
MCP bridge against a real (fake) stdio MCP server, and /v1/roblox/*.

Nothing here loads a real model or needs Roblox Studio."""

import copy
import json
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from zenit.inference.base import Engine, GenerationEvent, GenerationParams
from zenit.inference.hf_engine import HFEngine
from zenit.inference.registry import ModelPool
from zenit.persona.identity import Persona
from zenit.tools.agent import (CUT_OFF_MESSAGE, EVENT_OUTPUT_LIMIT, INVALID_TOOL, MODEL_OUTPUT_LIMIT, ParseError,
                               RobloxAgent, ToolCall, ToolCallFilter, parse_tool_calls)
from zenit.tools.roblox import (ROBLOX_SYSTEM_PROMPT, TOOL_NAMES, ArgumentError, RobloxService,
                                looks_like_luau_error, parse_studios, prepare_arguments, tool_specs)

HERE = Path(__file__).resolve().parent


def call(name, **arguments):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


# ------------------------------------------------------------------ fakes
class ScriptedEngine(Engine):
    """Plays back scripted model outputs in 3-character tokens."""

    def __init__(self, model_id="fake-tools", script=None, supports_tools=True, generation=None):
        self.id = model_id
        self.supports_tools = supports_tools
        self.context_window = 8192
        self.loaded = True
        self.card = {"version": "test", "base_model": "Fake/Model-1B"}
        self.generation = dict(generation or {})
        self.script = list(script or [])
        self.calls = []
        self.delay = 0.0
        self.raise_on_stream = None
        self.finishes = []                 # finish_reason per call (default "stop")

    def load(self):
        pass

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def info(self):
        return {"id": self.id, "engine": "fake", "supports_tools": self.supports_tools}

    def stream(self, messages, params, cancel=None, tools=None):
        self.calls.append({"messages": copy.deepcopy(messages), "params": params, "tools": tools})
        if self.raise_on_stream:
            raise self.raise_on_stream
        out = self.script.pop(0) if self.script else "Done."
        n = 0
        for i in range(0, len(out), 3):
            if cancel is not None and cancel.is_set():
                yield GenerationEvent("done", finish_reason="cancelled", prompt_tokens=10, completion_tokens=n)
                return
            if self.delay:
                time.sleep(self.delay)
            n += 1
            yield GenerationEvent("token", out[i:i + 3])
        reason = self.finishes.pop(0) if self.finishes else "stop"
        yield GenerationEvent("done", finish_reason=reason, prompt_tokens=10, completion_tokens=n)


class FakeBridge:
    def __init__(self, missing=("start_stop_play",)):
        self.connected = False
        self.closed = False
        self.fail_connect = False
        self.delay = 0.0
        self.calls = []
        self.studios = [{"id": "studio-aaa", "name": "Baseplate"}, {"id": "studio-bbb", "name": "Obby"}]
        self.names = ["list_roblox_studios", "set_active_studio"] + [n for n in TOOL_NAMES if n not in missing]

    def connect(self, timeout=30.0):
        if self.fail_connect:
            raise RuntimeError("Studio MCP is not running")
        self.connected = True

    def close(self):
        self.closed = True
        self.connected = False

    def list_tools(self, timeout=30.0):
        return [{"name": n, "description": "", "input_schema": {}} for n in self.names]

    def call_tool(self, name, arguments=None, timeout=120.0):
        args = dict(arguments or {})
        self.calls.append((name, args))
        if name == "list_roblox_studios":
            return True, json.dumps({"studios": self.studios})
        if name == "execute_luau":
            if self.delay:
                time.sleep(self.delay)
            if "error(" in args.get("code", ""):
                return False, "Luau error: boom"
            return True, "Built RedCube: 1 part"
        return True, f"{name} ok"


def executor_from(fn):
    log = []

    def run(name, args):
        log.append((name, args))
        return fn(name, args)
    run.log = log
    return run


def collect(gen):
    return list(gen)


# ------------------------------------------------------------- parsing
def test_parse_multiple_calls_and_prose():
    text = ("Let me look first.\n" + call("search_game_tree", path="Workspace") + "\n" +
            call("execute_luau", code='print("hi")') + "\n")
    prose, items = parse_tool_calls(text)
    assert prose == "Let me look first."
    assert [type(i) for i in items] == [ToolCall, ToolCall]
    assert items[0].name == "search_game_tree" and items[0].arguments == {"path": "Workspace"}
    assert items[1].arguments["code"] == 'print("hi")'
    assert items[0].id != items[1].id


def test_parse_tolerates_trailing_junk_and_string_arguments():
    text = '<tool_call>\n{"name": "inspect_instance", "arguments": {"path": "Workspace.A"}} }}junk\n</tool_call>'
    _, items = parse_tool_calls(text)
    assert isinstance(items[0], ToolCall) and items[0].arguments == {"path": "Workspace.A"}
    text = '<tool_call>{"name": "script_grep", "arguments": "{\\"query\\": \\"Touched\\"}"}</tool_call>'
    _, items = parse_tool_calls(text)
    assert items[0].arguments == {"query": "Touched"}


def test_parse_bad_json_is_reported_not_raised():
    text = "Trying.<tool_call>{'name': 'execute_luau', 'arguments': {'code': 'x'}}</tool_call>" + call("get_console_output")
    prose, items = parse_tool_calls(text)
    assert prose == "Trying."
    assert isinstance(items[0], ParseError) and "invalid JSON" in items[0].message
    assert isinstance(items[1], ToolCall) and items[1].name == "get_console_output"
    _, items = parse_tool_calls('<tool_call>{"arguments": {}}</tool_call>')
    assert isinstance(items[0], ParseError) and "name" in items[0].message
    _, items = parse_tool_calls('<tool_call>{"name": "x", "arguments": [1]}</tool_call>')
    assert isinstance(items[0], ParseError)


def test_parse_unclosed_final_block():
    _, items = parse_tool_calls('ok <tool_call>\n{"name": "get_studio_state", "arguments": {}}\n')
    assert isinstance(items[0], ToolCall) and items[0].name == "get_studio_state"
    _, items = parse_tool_calls('ok <tool_call>\n{"name": "execute_luau", "arguments": {"code": "pri')
    assert isinstance(items[0], ParseError)


def test_parse_accepts_raw_newlines_and_tabs_inside_strings():
    text = ('<tool_call>\n{"name": "execute_luau", "arguments": {"code": "local p = Instance.new(\\"Part\\")\n'
            '\tp.Parent = workspace"}}\n</tool_call>')
    _, items = parse_tool_calls(text)
    assert isinstance(items[0], ToolCall)
    assert items[0].arguments["code"] == 'local p = Instance.new("Part")\n\tp.Parent = workspace'
    _, items = parse_tool_calls('<tool_call>{"name": "execute_luau", "arguments": "{\\"code\\": \\"a\nb\\"}"}'
                                '</tool_call>')
    assert isinstance(items[0], ToolCall) and items[0].arguments == {"code": "a\nb"}


def test_parse_marks_unclosed_block():
    _, items = parse_tool_calls('x <tool_call>{"name": "execute_luau", "arguments": {"code": "pri')
    assert isinstance(items[0], ParseError) and items[0].unclosed
    _, items = parse_tool_calls(call("get_studio_state"))
    assert not items[0].unclosed


def feed_all(pieces):
    f = ToolCallFilter()
    out = "".join(f.feed(p) for p in pieces)
    return out + f.flush()


def test_filter_holds_back_tool_calls_split_across_tokens():
    text = "Building now.\n" + call("execute_luau", code="x") + "\n\nAll done."
    for size in (1, 2, 3, 5, 7, 100):
        pieces = [text[i:i + size] for i in range(0, len(text), size)]
        shown = feed_all(pieces)
        assert "<tool" not in shown and "execute_luau" not in shown and "</tool" not in shown
        assert shown.startswith("Building now.") and shown.endswith("All done.")


def test_filter_keeps_prose_around_a_block_apart():
    block = '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
    for size in (1, 2, 3, 7, 100):
        def split(t):
            return [t[i:i + size] for i in range(0, len(t), size)]
        assert feed_all(split("A " + block + "\n\nB text after")) == "A \n\nB text after"
        assert feed_all(split("x" + block + block + "   tail")) == "x\n\ntail"
        assert feed_all(split(block + "\nOnly after")) == "Only after"
        assert feed_all(split("pre " + block + " \n " + block)) == "pre "


def test_filter_releases_text_that_only_looks_like_a_tag():
    assert feed_all(["if a <", "b then", " end"]) == "if a <b then end"
    assert feed_all(["x <tool", "box"]) == "x <toolbox"
    assert feed_all(["trailing <tool"]) == "trailing <tool"
    f = ToolCallFilter()
    assert f.feed("Hi <tool_") == "Hi "            # held back until we know
    assert f.feed("call>{") == ""
    assert f.flush() == ""                           # unfinished block is never shown


# ---------------------------------------------------- argument mapping
def test_studio_id_and_datamodel_injection():
    a = prepare_arguments("execute_luau", {"code": "print(1)", "studio_id": "evil"}, "studio-aaa")
    assert a == {"code": "print(1)", "datamodel_type": "Edit", "studio_id": "studio-aaa"}
    a = prepare_arguments("execute_luau", {"code": "x", "datamodel_type": "server"}, "s1")
    assert a["datamodel_type"] == "Server"
    a = prepare_arguments("search_game_tree", {"path": "Workspace", "max_depth": "2", "bogus": 1}, "s1")
    assert a == {"path": "Workspace", "max_depth": 2, "head_limit": 60, "datamodel_type": "Edit", "studio_id": "s1"}
    a = prepare_arguments("multi_edit", {"file_path": "game.ServerScriptService.M", "datamodel_type": "Server",
                                         "edits": {"old_string": "", "new_string": "print(1)"}}, "s1")
    assert a["datamodel_type"] == "Edit" and isinstance(a["edits"], list)
    a = prepare_arguments("inspect_instance", {"path": "Workspace.A"}, "s1")
    assert "datamodel_type" not in a and a["studio_id"] == "s1"
    assert prepare_arguments("insert_asset", {"assetId": 12345}, "s1")["assetId"] == "12345"
    assert prepare_arguments("start_stop_play", {"is_start": "false"}, "s1")["is_start"] is False
    assert prepare_arguments("script_read", {"target_file": "game.X"}, "s1")["should_read_entire_file"] is True
    r = prepare_arguments("script_read", {"target_file": "game.X", "start_line_one_indexed": 1,
                                          "end_line_one_indexed_inclusive": 9}, "s1")
    assert r["should_read_entire_file"] is False
    with pytest.raises(ArgumentError):
        prepare_arguments("execute_luau", {}, "s1")
    with pytest.raises(ArgumentError):
        prepare_arguments("delete_everything", {}, "s1")


def test_common_argument_slips_are_coerced():
    assert prepare_arguments("execute_luau", {"code": "```lua\nprint(1)\nprint(2)\n```"}, "s")["code"] == \
        "print(1)\nprint(2)"
    assert prepare_arguments("execute_luau", {"code": "```\nprint(1)```"}, "s")["code"] == "print(1)"
    assert prepare_arguments("execute_luau", {"code": "x = '```'"}, "s")["code"] == "x = '```'"
    a = prepare_arguments("multi_edit", {"file_path": "game.ServerScriptService.Door", "className": "Script",
                                         "edits": '[{"new_string": "print(1)\nprint(2)"}]'}, "s")
    assert a["edits"] == [{"old_string": "", "new_string": "print(1)\nprint(2)"}]
    with pytest.raises(ArgumentError):
        prepare_arguments("multi_edit", {"file_path": "game.X", "edits": "[{not json"}, "s")
    a = prepare_arguments("search_game_tree", {"keywords": ["door", "wall"], "max_depth": 20}, "s")
    assert a["keywords"] == "door, wall" and a["max_depth"] == 10 and "path" not in a
    a = prepare_arguments("search_game_tree", {}, "s")
    assert a["path"] == "Workspace" and a["max_depth"] == 2 and a["head_limit"] == 60
    assert prepare_arguments("search_asset", {"query": "tree", "maxResults": 50}, "s")["maxResults"] == 20
    assert prepare_arguments("search_asset", {"query": "tree", "maxResults": 0}, "s")["maxResults"] == 1
    assert prepare_arguments("insert_asset", {"assetId": "rbxassetid://12345"}, "s")["assetId"] == "12345"
    r = prepare_arguments("script_read", {"target_file": "game.X", "start_line_one_indexed": "10"}, "s")
    assert r["should_read_entire_file"] is False and r["end_line_one_indexed_inclusive"] == 210
    r = prepare_arguments("script_read", {"target_file": "game.X", "end_line_one_indexed_inclusive": 40}, "s")
    assert r["start_line_one_indexed"] == 1 and r["should_read_entire_file"] is False
    assert prepare_arguments("execute_luau", {"code": "x"}, "s", "Server")["datamodel_type"] == "Server"


def test_luau_errors_reported_as_success_are_detected():
    assert looks_like_luau_error('[string "x"]:3: attempt to index nil with \'Parent\'')
    assert looks_like_luau_error("Workspace.Script:5: Expected 'end' (to close 'function' at line 1), got <eof>")
    assert looks_like_luau_error("Error: Color is not a valid member of Model")
    assert looks_like_luau_error("Stack Begin\nScript 'x', Line 2\nStack End")
    for fine in ("Built House: 42 parts at 0, 0, 0", "Errors: 0", "", "ran 8 chars"):
        assert not looks_like_luau_error(fine)


def test_system_prompt_teaches_the_build_rules():
    p = ROBLOX_SYSTEM_PROMPT
    for needle in ("math.rad", "PivotTo", "PrimaryPart", "y = h/2", "~=", "task.wait()", "Destroy()",
                   "ServerScriptService", "Parent last", "playtest"):
        assert needle in p, needle


def test_tool_specs_are_small_and_hide_studio_id():
    specs = tool_specs()
    assert [s["function"]["name"] for s in specs] == TOOL_NAMES
    blob = json.dumps(specs)
    assert "studio_id" not in blob
    assert len(blob) < 6000
    assert all(len(s["function"]["description"]) <= 120 for s in specs)


def test_parse_studios():
    assert parse_studios('{"studios":[{"id":"studio-74bd40fa","name":"Place1"}]}') == [
        {"id": "studio-74bd40fa", "name": "Place1"}]
    assert parse_studios("not json") == []
    assert parse_studios('{"studios": []}') == []


def test_service_picks_studio_and_injects_it():
    bridge = FakeBridge()
    svc = RobloxService({"enabled": True}, lambda cfg: bridge)
    st = svc.status()
    assert st["connected"] and st["studio_id"] == "studio-aaa" and len(st["studios"]) == 2
    assert "start_stop_play" not in st["tools"] and "execute_luau" in st["tools"]
    ok, out = svc.call("execute_luau", {"code": "print(1)"})
    assert ok and bridge.calls[-1] == ("execute_luau", {"code": "print(1)", "datamodel_type": "Edit",
                                                        "studio_id": "studio-aaa"})
    assert svc.set_studio("studio-bbb") and svc.studio_id == "studio-bbb"
    assert not svc.set_studio("studio-zzz") and svc.studio_id == "studio-bbb"
    ok, out = svc.call("start_stop_play", {"is_start": True})     # not exposed by this server
    assert not ok and "unknown tool" in out
    ok, out = svc.call("execute_luau", {})
    assert not ok and out.startswith("Error:")
    preferred = RobloxService({"enabled": True, "studio_id": "studio-bbb"}, lambda cfg: FakeBridge())
    assert preferred.status()["studio_id"] == "studio-bbb"
    svc.close()
    assert bridge.closed


def test_service_call_uses_the_pinned_studio():
    bridge = FakeBridge()
    svc = RobloxService({"enabled": True}, lambda cfg: bridge)
    svc.status()
    assert svc.studio_id == "studio-aaa"
    ok, _ = svc.call("execute_luau", {"code": "print(1)"}, "studio-bbb")
    assert ok and bridge.calls[-1][1]["studio_id"] == "studio-bbb"
    svc.studio_id = None                       # e.g. a status poll failed mid-run
    ok, _ = svc.call("execute_luau", {"code": "print(2)"}, "studio-bbb")
    assert ok and bridge.calls[-1][1]["studio_id"] == "studio-bbb"
    ok, out = svc.call("execute_luau", {"code": "print(3)"})
    assert not ok and "no Roblox Studio" in out


def test_service_tracks_playtests_and_flags_luau_errors():
    class Bridge(FakeBridge):
        def call_tool(self, name, arguments=None, timeout=120.0):
            if name == "execute_luau" and "nil" in (arguments or {}).get("code", ""):
                self.calls.append((name, dict(arguments)))
                return True, '[string "x"]:1: attempt to index nil with \'Parent\''
            return super().call_tool(name, arguments, timeout)
    bridge = Bridge(missing=())
    svc = RobloxService({"enabled": True}, lambda cfg: bridge)
    svc.status()
    assert svc.call("start_stop_play", {"is_start": "true"})[0] and svc.play_mode["studio-aaa"] is True
    svc.call("search_game_tree", {})
    assert bridge.calls[-1][1]["datamodel_type"] == "Server"
    svc.call("execute_luau", {"code": "x", "datamodel_type": "Client"})
    assert bridge.calls[-1][1]["datamodel_type"] == "Client"
    assert svc.call("start_stop_play", {"is_start": False})[0] and svc.play_mode["studio-aaa"] is False
    svc.call("execute_luau", {"code": "print(1)"})
    assert bridge.calls[-1][1]["datamodel_type"] == "Edit"
    ok, out = svc.call("execute_luau", {"code": "nil.Parent = 1"})
    assert ok is False and "attempt to index nil" in out


def test_service_reports_connect_failure_and_disabled():
    bridge = FakeBridge()
    bridge.fail_connect = True
    st = RobloxService({"enabled": True}, lambda cfg: bridge).status()
    assert st["available"] and not st["connected"] and "not running" in st["error"]
    st = RobloxService({"enabled": False}, lambda cfg: FakeBridge()).status()
    assert not st["available"] and not st["connected"] and "disabled" in st["error"]


# ---------------------------------------------------------- agent loop
PARAMS = GenerationParams(max_tokens=256)


def test_agent_streams_prose_and_runs_tools():
    engine = ScriptedEngine(script=[
        "I'll check the workspace first.\n" + call("search_game_tree", path="Workspace") + "\n" +
        call("execute_luau", code="print('built')"),
        "I built a red cube called RedCube in workspace.",
    ])
    ex = executor_from(lambda n, a: (True, f"{n} done"))
    agent = RobloxAgent(ex, tool_specs())
    events = collect(agent.run(engine, [{"role": "system", "content": "s"}, {"role": "user", "content": "cube"}],
                               PARAMS, threading.Event(), 8))
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done" and events[-1] == {"type": "done", "finish_reason": "stop", "steps": 2}
    prose = "".join(e["text"] for e in events if e["type"] == "token")
    assert "<tool_call>" not in prose and "search_game_tree" not in prose
    assert prose.startswith("I'll check the workspace first.") and prose.endswith("called RedCube in workspace.")
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert [c["name"] for c in calls] == ["search_game_tree", "execute_luau"]
    assert [r["id"] for r in results] == [c["id"] for c in calls] and all(r["ok"] for r in results)
    assert kinds.index("tool_call") < kinds.index("tool_result")
    assert ex.log == [("search_game_tree", {"path": "Workspace"}), ("execute_luau", {"code": "print('built')"})]
    second = engine.calls[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "tool"]
    assistant = second[2]
    assert assistant["content"] == "I'll check the workspace first."
    assert [c["function"]["name"] for c in assistant["tool_calls"]] == ["search_game_tree", "execute_luau"]
    assert assistant["tool_calls"][1]["function"]["arguments"] == {"code": "print('built')"}
    assert second[3]["content"] == "search_game_tree done"
    assert engine.calls[0]["tools"] and engine.calls[0]["tools"][0]["type"] == "function"
    assert len(agent.calls) == 2 and agent.prompt_tokens == 20


def test_agent_error_tool_result_goes_back_to_model():
    engine = ScriptedEngine(script=[call("execute_luau", code="error('x')"),
                                    "The script failed with an error, so nothing was built."])
    agent = RobloxAgent(executor_from(lambda n, a: (False, "Luau error: boom")), tool_specs())
    events = collect(agent.run(engine, [{"role": "user", "content": "go"}], PARAMS, None, 4))
    res = next(e for e in events if e["type"] == "tool_result")
    assert res["ok"] is False and "boom" in res["output"]
    tool_msg = engine.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"].startswith("Error") and "boom" in tool_msg["content"]
    assert events[-1]["finish_reason"] == "stop"


def test_agent_reports_parse_errors_so_the_model_can_retry():
    engine = ScriptedEngine(script=["<tool_call>{'name': 'execute_luau'}</tool_call>",
                                    call("execute_luau", code="print(1)"), "Fixed and done."])
    ex = executor_from(lambda n, a: (True, "1"))
    events = collect(RobloxAgent(ex, tool_specs()).run(engine, [{"role": "user", "content": "go"}],
                                                      PARAMS, None, 5))
    calls = [e for e in events if e["type"] == "tool_call"]
    assert calls[0]["name"] == INVALID_TOOL and calls[1]["name"] == "execute_luau"
    first_result = next(e for e in events if e["type"] == "tool_result")
    assert first_result["ok"] is False and "could not parse" in first_result["output"]
    retry_prompt = engine.calls[1]["messages"]
    assert retry_prompt[-2]["role"] == "assistant" and "{'name'" in retry_prompt[-2]["content"]
    assert "tool_calls" not in retry_prompt[-2]
    assert retry_prompt[-1]["role"] == "tool" and "could not parse" in retry_prompt[-1]["content"]
    assert ex.log == [("execute_luau", {"code": "print(1)"})]
    assert events[-1] == {"type": "done", "finish_reason": "stop", "steps": 3}


def test_agent_stops_at_max_steps():
    engine = ScriptedEngine(script=[call("get_console_output")] * 10)
    events = collect(RobloxAgent(executor_from(lambda n, a: (True, "")), tool_specs())
                     .run(engine, [{"role": "user", "content": "loop"}], PARAMS, None, 3))
    assert events[-1] == {"type": "done", "finish_reason": "max_steps", "steps": 3}
    assert len(engine.calls) == 3
    assert sum(e["type"] == "tool_result" for e in events) == 3


def test_agent_cancel_between_tools():
    cancel = threading.Event()

    def ex(name, args):
        cancel.set()                     # user presses stop while the first tool runs
        return True, "ok"
    engine = ScriptedEngine(script=[call("execute_luau", code="a") + call("execute_luau", code="b"), "never"])
    agent = RobloxAgent(executor_from(ex), tool_specs())
    events = collect(agent.run(engine, [{"role": "user", "content": "go"}], PARAMS, cancel, 8))
    assert events[-1]["finish_reason"] == "cancelled"
    assert sum(e["type"] == "tool_call" for e in events) == 1
    assert len(engine.calls) == 1


def test_agent_cancel_during_generation():
    cancel = threading.Event()
    cancel.set()
    engine = ScriptedEngine(script=["Long answer " * 20])
    events = collect(RobloxAgent(executor_from(lambda n, a: (True, "")), tool_specs())
                     .run(engine, [{"role": "user", "content": "go"}], PARAMS, cancel, 8))
    assert events[-1] == {"type": "done", "finish_reason": "cancelled", "steps": 1}


def test_agent_truncates_tool_output():
    engine = ScriptedEngine(script=[call("script_read", target_file="game.Big"), "Read it."])
    big = "x" * 20_000
    events = collect(RobloxAgent(executor_from(lambda n, a: (True, big)), tool_specs())
                     .run(engine, [{"role": "user", "content": "read"}], PARAMS, None, 4))
    res = next(e for e in events if e["type"] == "tool_result")
    assert EVENT_OUTPUT_LIMIT <= len(res["output"]) < EVENT_OUTPUT_LIMIT + 60 and "truncated" in res["output"]
    fed = engine.calls[1]["messages"][-1]["content"]
    assert MODEL_OUTPUT_LIMIT <= len(fed) < MODEL_OUTPUT_LIMIT + 60


def test_agent_explains_a_tool_call_cut_off_by_the_length_limit():
    engine = ScriptedEngine(script=['Building.<tool_call>\n{"name": "execute_luau", "arguments": {"code": "local a',
                                    "Sorry, I will split it."])
    engine.finishes = ["length"]
    events = collect(RobloxAgent(executor_from(lambda n, a: (True, "")), tool_specs())
                     .run(engine, [{"role": "user", "content": "big castle"}], PARAMS, None, 4))
    res = next(e for e in events if e["type"] == "tool_result")
    assert res["ok"] is False and res["output"] == CUT_OFF_MESSAGE
    assert engine.calls[1]["messages"][-1]["content"] == CUT_OFF_MESSAGE
    # a malformed but complete reply still gets the JSON hint
    engine = ScriptedEngine(script=["<tool_call>{'name': 'x'}", "ok"])
    engine.finishes = ["stop"]
    events = collect(RobloxAgent(executor_from(lambda n, a: (True, "")), tool_specs())
                     .run(engine, [{"role": "user", "content": "go"}], PARAMS, None, 4))
    assert "could not parse" in next(e for e in events if e["type"] == "tool_result")["output"]


def test_agent_keeps_the_end_of_console_output():
    engine = ScriptedEngine(script=[call("get_console_output"), "Found the error."])
    log = "old line\n" * 3000 + "Workspace.Script:3: attempt to call a nil value"
    events = collect(RobloxAgent(executor_from(lambda n, a: (True, log)), tool_specs())
                     .run(engine, [{"role": "user", "content": "errors?"}], PARAMS, None, 4))
    res = next(e for e in events if e["type"] == "tool_result")
    assert res["output"].endswith("attempt to call a nil value") and "truncated" in res["output"]
    assert engine.calls[1]["messages"][-1]["content"].endswith("attempt to call a nil value")


def test_agent_engine_error_ends_with_error_event():
    engine = ScriptedEngine()
    engine.raise_on_stream = RuntimeError("CUDA out of memory")
    events = collect(RobloxAgent(executor_from(lambda n, a: (True, "")), tool_specs())
                     .run(engine, [{"role": "user", "content": "go"}], PARAMS, None, 4))
    assert events[0]["type"] == "error" and "out of memory" in events[0]["message"]
    assert events[-1] == {"type": "done", "finish_reason": "error", "steps": 1}


# ------------------------------------------------ engines, pool, persona
def test_fold_roles_keeps_tool_turns_in_order():
    msgs = [{"role": "system", "content": "sys"}, {"role": "memory", "content": "likes red"},
            {"role": "user", "content": "build"},
            {"role": "assistant", "content": "ok", "tool_calls": [
                {"type": "function", "function": {"name": "execute_luau", "arguments": {"code": "x"}}}]},
            {"role": "tool", "name": "execute_luau", "content": "done"},
            {"role": "assistant", "content": "Built."}]
    out = HFEngine._fold_roles(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "assistant"]
    assert "<memory>" in out[0]["content"]
    assert out[2]["tool_calls"][0]["function"]["name"] == "execute_luau"
    assert "tool_calls" not in out[4]


def test_pool_resolves_relative_paths_and_passes_extras(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "models" / "local-model").mkdir(parents=True)
    (tmp_path / "configs" / "serve.json").write_text(json.dumps({"default_model": "local", "models": [
        {"id": "local", "engine": "hf", "path": "models/local-model", "supports_tools": True,
         "card": {"version": "2.0-test", "base_model": "Org/Base"}, "generation": {"top_k": 20}},
        {"id": "hub", "engine": "hf", "path": "Qwen/Qwen3-4B-Base"},
        {"id": "nano", "engine": "native", "path": "models/missing"}]}), encoding="utf-8")
    pool = ModelPool(tmp_path / "configs" / "serve.json", tmp_path / "models", root=tmp_path)
    local = pool.engines["local"]
    assert Path(local.path) == tmp_path / "models" / "local-model"
    assert local.supports_tools and local.generation == {"top_k": 20} and local.card["version"] == "2.0-test"
    assert pool.engines["hub"].path == "Qwen/Qwen3-4B-Base" and not pool.engines["hub"].supports_tools
    assert pool.engines["nano"].supports_tools is False
    listed = {m["id"]: m for m in pool.list()}
    assert listed["local"]["version"] == "2.0-test" and listed["local"]["supports_tools"]
    assert not listed["local"]["loaded"]              # listing never loads weights


def test_repo_serve_config_defaults_to_qwen():
    cfg = json.loads((HERE.parent / "configs" / "serve.json").read_text(encoding="utf-8"))
    ids = {m["id"]: m for m in cfg["models"]}
    assert cfg["default_model"] == "zenit-qwen3-4b" and ids["zenit-qwen3-4b"]["supports_tools"]
    assert "zenit-0.1-nano" in ids and ids["zenit-2.0-sft"]["enabled"] is False


def test_persona_sampling_uses_model_defaults_only_where_profile_is_default():
    p = Persona()
    gen = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}
    bal = p.sampling("balanced", gen)
    assert (bal["top_p"], bal["top_k"], bal["repetition_penalty"]) == (0.8, 20, 1.05)
    precise = p.sampling("precise", gen)
    assert precise["temperature"] == 0.2 and precise["top_p"] == 0.8 and precise["top_k"] == 20
    creative = p.sampling("creative", gen)
    assert creative["temperature"] == 0.95 and creative["top_p"] == 0.95 and creative["top_k"] == 20
    assert p.sampling("balanced", None)["top_k"] == 50


def test_card_without_fine_tuning_is_described_honestly():
    text = Persona().system_for(None, {"version": "2.0-qwen3-4b", "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                                       "fine_tuned": False})
    assert "not trained or fine-tuned by the ZENIT project" in text


# ------------------------------------------------ real MCP over stdio
def test_mcp_bridge_talks_to_a_real_stdio_server():
    pytest.importorskip("mcp")
    from zenit.tools.mcp_bridge import MCPBridge, MCPConnectError
    bridge = MCPBridge(sys.executable, [str(HERE / "fake_mcp_server.py")])
    try:
        bridge.connect(60)
        assert bridge.connected
        tools = {t["name"]: t for t in bridge.list_tools()}
        assert {"list_roblox_studios", "execute_luau"} <= set(tools)
        assert "code" in tools["execute_luau"]["input_schema"]["properties"]
        ok, text = bridge.call_tool("list_roblox_studios", {})
        assert ok and parse_studios(text)[0]["id"] == "studio-test1"
        ok, text = bridge.call_tool("execute_luau", {"code": "print(1)", "datamodel_type": "Edit",
                                                     "studio_id": "studio-test1"})
        assert ok and text == "ran 8 chars in Edit on studio-test1"
        ok, text = bridge.call_tool("fail_tool", {"studio_id": "x"})
        assert not ok
        # The server dies mid-call: the call is reported (not retried), the next call reconnects.
        ok, text = bridge.call_tool("crash", {}, timeout=30)
        assert not ok
        ok, text = bridge.call_tool("list_roblox_studios", {}, timeout=60)
        assert ok and "studio-test2" in text
        # The server exits while idle: the bridge notices, and the next call starts a new one.
        ok, text = bridge.call_tool("crash_soon", {}, timeout=30)
        assert ok and text == "bye"
        deadline = time.time() + 15
        while bridge.connected and time.time() < deadline:
            time.sleep(0.1)
        assert not bridge.connected
        ok, text = bridge.call_tool("list_roblox_studios", {}, timeout=60)
        assert ok and "studio-test1" in text
    finally:
        t0 = time.time()
        bridge.close()
        assert time.time() - t0 < 15
    assert not bridge.connected
    broken = MCPBridge("zenit-no-such-command-xyz", [])
    with pytest.raises(MCPConnectError):
        broken.connect(20)
    broken.close()


# --------------------------------------------------------- API endpoints
requests = pytest.importorskip("requests")
uvicorn = pytest.importorskip("uvicorn")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def rserver(tmp_path_factory):
    from zenit.api.app import create_app
    from zenit.api.auth import create_key
    root = tmp_path_factory.mktemp("zenit_roblox")
    (root / "configs").mkdir()
    (root / "web" / "static").mkdir(parents=True)
    (root / "configs" / "serve.json").write_text(json.dumps(
        {"default_model": "nano", "models": [{"id": "nano", "engine": "native", "path": "models/none"}]}),
        encoding="utf-8")
    bridge = FakeBridge()
    app = create_app(root, roblox_bridge=bridge, roblox_config={"enabled": True, "max_steps": 6})
    s = app.state.zenit
    engine = ScriptedEngine(generation={"temperature": 0.33, "top_k": 7})
    s.pool.engines[engine.id] = engine
    s.pool.specs[engine.id] = {"id": engine.id, "engine": "fake"}
    s.pool.default = engine.id
    user = create_key(s.db, "builder", "user", rate_per_min=1000)
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}", "key": user["key"], "engine": engine, "bridge": bridge, "server": s}
    srv.should_exit = True
    th.join(timeout=10)
    assert bridge.closed                      # app shutdown closes the MCP bridge
    shutil.rmtree(root, ignore_errors=True)


def h(key):
    return {"Authorization": f"Bearer {key}"}


def sse_events(resp):
    out = []
    for line in resp.iter_lines(chunk_size=1, decode_unicode=True):
        if line and line.startswith("data: "):
            body = line[6:]
            out.append(body if body == "[DONE]" else json.loads(body))
    return out


def test_roblox_status_and_studio_selection(rserver):
    url, key = rserver["url"], rserver["key"]
    assert requests.get(url + "/v1/roblox/status").status_code == 401
    st = requests.get(url + "/v1/roblox/status", headers=h(key)).json()
    assert st["available"] and st["connected"] and st["error"] is None
    assert st["studios"] == rserver["bridge"].studios and st["studio_id"] == "studio-aaa"
    assert "execute_luau" in st["tools"] and "studio_id" not in json.dumps(st["tools"])
    assert st["model_supports_tools"] is True
    r = requests.post(url + "/v1/roblox/studio", json={"studio_id": "studio-bbb"}, headers=h(key))
    assert r.status_code == 200 and r.json()["studio_id"] == "studio-bbb"
    r = requests.post(url + "/v1/roblox/studio", json={"studio_id": "studio-nope"}, headers=h(key))
    assert r.status_code == 404 and r.json()["error"]["type"] == "invalid_request_error"
    assert requests.post(url + "/v1/roblox/studio", json={}, headers=h(key)).status_code == 422
    requests.post(url + "/v1/roblox/studio", json={"studio_id": "studio-aaa"}, headers=h(key))


def test_roblox_chat_streams_contract_events_and_saves_conversation(rserver):
    url, key, engine, bridge = rserver["url"], rserver["key"], rserver["engine"], rserver["bridge"]
    engine.script = ["Building a red cube.\n" + call("execute_luau", code="local m = Instance.new('Model')"),
                     "Done: RedCube is in workspace."]
    bridge.calls.clear()
    r = requests.post(url + "/v1/roblox/chat", json={"message": "build a red cube"}, headers=h(key),
                      stream=True, timeout=60)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r)
    assert events[-1] == "[DONE]"
    events = events[:-1]
    meta = events[0]
    assert meta["type"] == "meta" and meta["model"] == "fake-tools" and meta["studio_id"] == "studio-aaa"
    assert meta["generation_id"].startswith("gen_") and meta["conversation_id"]
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done" and events[-1]["finish_reason"] == "stop" and events[-1]["steps"] == 2
    assert kinds.index("tool_call") < kinds.index("tool_result")
    tc = next(e for e in events if e["type"] == "tool_call")
    assert tc["name"] == "execute_luau" and tc["arguments"] == {"code": "local m = Instance.new('Model')"}
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["id"] == tc["id"] and tr["ok"] and tr["output"] == "Built RedCube: 1 part"
    prose = "".join(e["text"] for e in events if e["type"] == "token")
    assert "<tool_call>" not in prose and "RedCube is in workspace" in prose
    luau = [c for c in bridge.calls if c[0] == "execute_luau"]
    assert luau[0][1]["studio_id"] == "studio-aaa" and luau[0][1]["datamodel_type"] == "Edit"
    # build mode samples with configs/roblox.json "generation", not the chat defaults
    p = engine.calls[-1]["params"]
    assert p.temperature == 0.3 and p.top_k == 20 and p.repetition_penalty == 1.0
    first_prompt = engine.calls[-2]["messages"]
    assert first_prompt[0]["role"] == "system" and "Roblox" in first_prompt[0]["content"]
    assert first_prompt[-1] == {"role": "user", "content": "build a red cube"}

    cid = meta["conversation_id"]
    conv = requests.get(f"{url}/v1/conversations/{cid}", headers=h(key)).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["title"] == "build a red cube"
    asst = conv["messages"][1]
    assert "RedCube is in workspace" in asst["content"] and "<tool_call>" not in asst["content"]
    assert asst["meta"]["roblox"] is True and asst["meta"]["generation_id"] == meta["generation_id"]
    saved = asst["meta"]["tool_calls"]
    assert saved == [{"name": "execute_luau", "arguments": {"code": "local m = Instance.new('Model')"},
                      "ok": True, "output": "Built RedCube: 1 part"}]

    # follow-up in the same conversation sees the earlier turn and its tools
    engine.script = ["Made it blue."]
    r = requests.post(url + "/v1/roblox/chat", json={"message": "make it blue", "conversation_id": cid},
                      headers=h(key), stream=True, timeout=60)
    events = sse_events(r)
    assert events[-2]["type"] == "done" and events[0]["conversation_id"] == cid
    prompt = engine.calls[-1]["messages"]
    assert [m["role"] for m in prompt] == ["system", "user", "assistant", "user"]
    assert "(Earlier tool results: execute_luau ok: Built RedCube: 1 part)" in prompt[2]["content"]
    conv = requests.get(f"{url}/v1/conversations/{cid}", headers=h(key)).json()
    assert len(conv["messages"]) == 4 and conv["messages"][3]["meta"]["tool_calls"] == []


def test_roblox_chat_cancel(rserver):
    url, key, engine = rserver["url"], rserver["key"], rserver["engine"]
    engine.script = ["This is a very long explanation. " * 30]
    engine.delay = 0.02
    try:
        r = requests.post(url + "/v1/roblox/chat", json={"message": "explain"}, headers=h(key),
                          stream=True, timeout=60)
        events = []
        for line in r.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            ev = json.loads(body)
            events.append(ev)
            if ev["type"] == "token" and sum(e["type"] == "token" for e in events) == 2:
                gid = events[0]["generation_id"]
                c = requests.post(f"{url}/v1/generations/{gid}/cancel", headers=h(key)).json()
                assert c == {"cancelled": True}
        assert events[-1]["type"] == "done" and events[-1]["finish_reason"] == "cancelled"
        assert sum(e["type"] == "token" for e in events) < 100
    finally:
        engine.delay = 0.0


def test_roblox_chat_stop_still_saves_the_running_tool_call(rserver):
    """The UI's Stop cancels and drops the stream while execute_luau runs in Studio:
    the call that finished afterwards must still be saved with the turn."""
    url, key, engine, bridge = rserver["url"], rserver["key"], rserver["engine"], rserver["bridge"]
    engine.script = ["Building a house.\n" + call("execute_luau", code="print('house')"), "Built it."]
    bridge.delay = 1.0
    try:
        r = requests.post(url + "/v1/roblox/chat", json={"message": "build a house"}, headers=h(key),
                          stream=True, timeout=60)
        cid = None
        for line in r.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "meta":
                cid, gid = ev["conversation_id"], ev["generation_id"]
            if ev["type"] == "tool_call":
                requests.post(f"{url}/v1/generations/{gid}/cancel", headers=h(key))
                break
        r.close()
        msgs = []
        deadline = time.time() + 15
        while time.time() < deadline:
            msgs = requests.get(f"{url}/v1/conversations/{cid}", headers=h(key)).json()["messages"]
            if len(msgs) == 2:
                break
            time.sleep(0.2)
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        meta = msgs[1]["meta"]
        assert meta["finish_reason"] == "cancelled" and meta["steps"] >= 1
        assert [c["name"] for c in meta["tool_calls"]] == ["execute_luau"] and meta["tool_calls"][0]["ok"]
        assert msgs[1]["content"] == "Building a house."
    finally:
        bridge.delay = 0.0


def test_roblox_chat_stops_a_playtest_it_started(rserver):
    url, key, engine, bridge, s = (rserver["url"], rserver["key"], rserver["engine"], rserver["bridge"],
                                   rserver["server"])
    saved_tools = s.roblox.server_tools
    s.roblox.server_tools = set(saved_tools if saved_tools is not None else TOOL_NAMES) | {"start_stop_play"}
    engine.script = [call("start_stop_play", is_start=True), "Playtest started."]
    bridge.calls.clear()
    try:
        r = requests.post(url + "/v1/roblox/chat", json={"message": "playtest it"}, headers=h(key),
                          stream=True, timeout=60)
        events = sse_events(r)
        assert events[-2]["type"] == "done"
        plays = [c[1]["is_start"] for c in bridge.calls if c[0] == "start_stop_play"]
        assert plays == [True, False] and not s.roblox.play_mode.get("studio-aaa")
    finally:
        s.roblox.server_tools = saved_tools
        s.roblox.play_mode.clear()


def test_roblox_chat_rejects_bad_requests(rserver):
    url, key, bridge = rserver["url"], rserver["key"], rserver["bridge"]
    r = requests.post(url + "/v1/roblox/chat", json={"message": "x", "model": "nano"}, headers=h(key))
    assert r.status_code == 400 and "tool" in r.json()["error"]["message"]
    r = requests.post(url + "/v1/roblox/chat", json={"message": "x", "model": "ghost"}, headers=h(key))
    assert r.status_code == 404
    assert requests.post(url + "/v1/roblox/chat", json={"message": ""}, headers=h(key)).status_code == 422
    assert requests.post(url + "/v1/roblox/chat", json={"message": "x", "max_steps": 17},
                         headers=h(key)).status_code == 422
    assert requests.post(url + "/v1/roblox/chat", json={"message": "x", "conversation_id": "nope"},
                         headers=h(key)).status_code == 404
    bridge.fail_connect, bridge.connected = True, False
    try:
        r = requests.post(url + "/v1/roblox/chat", json={"message": "x"}, headers=h(key))
        assert r.status_code == 503 and r.json()["error"]["type"] == "service_unavailable"
        st = requests.get(url + "/v1/roblox/status", headers=h(key)).json()
        assert st["available"] and not st["connected"] and "not running" in st["error"]
    finally:
        bridge.fail_connect = False
    assert requests.get(url + "/v1/roblox/status", headers=h(key)).json()["connected"]


def test_plain_chat_still_works_with_tools_engine(rserver):
    url, key, engine = rserver["url"], rserver["key"], rserver["engine"]
    engine.script = ["Hello from the fake model."]
    r = requests.post(url + "/v1/chat/completions", json={"message": "hi"}, headers=h(key), timeout=60)
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "Hello from the fake model."
    assert engine.calls[-1]["tools"] is None
    p = engine.calls[-1]["params"]
    assert p.temperature == 0.33 and p.top_k == 7 and p.top_p == 0.9     # balanced profile + model defaults
