"""Tool-calling loop for models that emit Qwen/Hermes style tool calls:

    <tool_call>
    {"name": "execute_luau", "arguments": {"code": "..."}}
    </tool_call>

Each step: generate with the tool list, stream prose to the user while
holding back everything inside <tool_call> blocks, parse the blocks, run
them, append the assistant turn (with tool_calls) and one tool message per
call, and generate again. The loop ends when a step makes no tool calls, on
cancel, or after max_steps.

RobloxAgent.run yields the event dicts of the /v1/roblox/chat contract
(everything except "meta"):
    {"type": "token", "text"}
    {"type": "tool_call", "id", "name", "arguments"}
    {"type": "tool_result", "id", "ok", "output"}
    {"type": "error", "message"}
    {"type": "done", "finish_reason", "steps"}
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterator

from ..inference.base import GenerationParams

OPEN, CLOSE = "<tool_call>", "</tool_call>"
MODEL_OUTPUT_LIMIT = 6000          # chars of tool output fed back to the model
EVENT_OUTPUT_LIMIT = 4000          # chars of tool output sent to the browser
INVALID_TOOL = "invalid_tool_call"


TAIL_TOOLS = {"get_console_output"}   # newest lines (the errors) are at the end
_JSON = json.JSONDecoder(strict=False)  # small models put raw newlines/tabs inside "code" strings
CUT_OFF_MESSAGE = ("Error: your tool call was cut off at the output limit. Split the build into several "
                   "smaller execute_luau calls (one section each, under about 80 lines).")


def truncate(text: str, limit: int, keep_tail: bool = False) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    if keep_tail:
        return f"...[truncated {len(text) - limit} earlier chars]\n" + text[-limit:]
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str = ""
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:12])
    unclosed: bool = False             # no </tool_call> (generation ended inside the block)


@dataclass
class ParseError:
    message: str
    raw: str
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:12])
    unclosed: bool = False


def _parse_block(body: str) -> ToolCall | ParseError:
    raw = body.strip()
    start = raw.find("{")
    if start < 0:
        return ParseError("no JSON object found", raw)
    try:
        obj, _end = _JSON.raw_decode(raw[start:])     # tolerates trailing junk and raw newlines
    except json.JSONDecodeError as err:
        return ParseError(f"invalid JSON ({err.msg} at char {err.pos}); use double quotes and escape "
                          "newlines inside strings as \\n", raw)
    if not isinstance(obj, dict):
        return ParseError("tool call must be a JSON object", raw)
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return ParseError('missing "name"', raw)
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args, strict=False) if args.strip() else {}
        except json.JSONDecodeError as err:
            return ParseError(f'"arguments" is a string that is not valid JSON ({err.msg})', raw)
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return ParseError('"arguments" must be a JSON object', raw)
    return ToolCall(name, args, raw)


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall | ParseError]]:
    """Split model output into prose (outside tool blocks) and the parsed
    calls in order. An unclosed final <tool_call> block (generation ended
    early) is parsed too."""
    prose: list[str] = []
    items: list[ToolCall | ParseError] = []
    pos = 0
    while True:
        i = text.find(OPEN, pos)
        if i < 0:
            prose.append(text[pos:])
            break
        prose.append(text[pos:i])
        j = text.find(CLOSE, i + len(OPEN))
        if j < 0:
            item = _parse_block(text[i + len(OPEN):])
            item.unclosed = True
            items.append(item)
            break
        items.append(_parse_block(text[i + len(OPEN):j]))
        pos = j + len(CLOSE)
    return _clean_prose("".join(prose)), items


def _clean_prose(text: str) -> str:
    text = text.replace(CLOSE, "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class ToolCallFilter:
    """Streams prose and holds back <tool_call> blocks. Handles tags split
    across tokens by buffering any suffix that could start a tag."""

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False
        self.after_block = False
        self.emitted = False               # non-whitespace prose already released in this step

    @staticmethod
    def _partial_suffix(text: str, tag: str) -> int:
        for n in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:n]):
                return n
        return 0

    def feed(self, piece: str) -> str:
        self.buf += piece
        out: list[str] = []
        while True:
            if self.inside:
                j = self.buf.find(CLOSE)
                if j < 0:
                    keep = self._partial_suffix(self.buf, CLOSE)
                    self.buf = self.buf[len(self.buf) - keep:] if keep else ""
                    break
                self.buf = self.buf[j + len(CLOSE):]
                self.inside, self.after_block = False, True
                continue
            if self.after_block:                       # collapse whitespace between/after blocks
                stripped = self.buf.lstrip()
                if not stripped:
                    self.buf = ""
                    break
                if len(stripped) < len(OPEN) and OPEN.startswith(stripped):
                    self.buf = stripped                # maybe another block: wait for more text
                    break
                self.buf = stripped
                self.after_block = False
                if (self.emitted or "".join(out).strip()) and not stripped.startswith(OPEN):
                    out.append("\n\n")                # keep prose before and after a block apart
            i = self.buf.find(OPEN)
            if i >= 0:
                out.append(self.buf[:i])
                self.buf = self.buf[i + len(OPEN):]
                self.inside = True
                continue
            keep = self._partial_suffix(self.buf, OPEN)
            out.append(self.buf[:len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]
            break
        shown = "".join(out)
        if shown.strip():
            self.emitted = True
        return shown

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        if rest and self.after_block:
            rest = rest.lstrip()
            if rest and self.emitted:
                rest = "\n\n" + rest
        self.buf = ""
        if rest.strip():
            self.emitted = True
        return rest


Executor = Callable[[str, dict], "tuple[bool, str]"]


class RobloxAgent:
    """Runs the generate -> tool -> generate loop. `executor(name, args)`
    performs one call and returns (ok, output); it must not raise."""

    def __init__(self, executor: Executor, tools: list[dict]) -> None:
        self.executor = executor
        self.tools = tools
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.prose: list[str] = []        # every prose segment streamed to the user
        self.calls: list[dict] = []       # {"id","name","arguments","ok","output"} for storage

    def run(self, engine, messages: list[dict], params: GenerationParams,
            cancel: threading.Event | None = None, max_steps: int = 8) -> Iterator[dict]:
        cancel = cancel or threading.Event()
        msgs = list(messages)
        steps = 0
        max_steps = max(1, int(max_steps))
        new_paragraph = False      # separate prose of consecutive steps
        while steps < max_steps:
            steps += 1
            filt = ToolCallFilter()
            raw: list[str] = []
            finish = "stop"
            try:
                for ev in engine.stream(msgs, params, cancel, tools=self.tools):
                    if ev.type == "token":
                        raw.append(ev.text)
                        shown = filt.feed(ev.text)
                        if shown and new_paragraph:
                            shown = shown.lstrip()
                            if shown:
                                shown, new_paragraph = "\n\n" + shown, False
                        if shown:
                            self.prose.append(shown)
                            yield {"type": "token", "text": shown}
                    else:
                        finish = ev.finish_reason or "stop"
                        self.prompt_tokens += ev.prompt_tokens
                        self.completion_tokens += ev.completion_tokens
            except Exception as err:
                yield {"type": "error", "message": f"engine error: {type(err).__name__}: {err}"}
                yield {"type": "done", "finish_reason": "error", "steps": steps}
                return
            tail = filt.flush()
            if tail and new_paragraph:
                tail = tail.lstrip()
                if tail:
                    tail, new_paragraph = "\n\n" + tail, False
            if tail:
                self.prose.append(tail)
                yield {"type": "token", "text": tail}
            if cancel.is_set() or finish == "cancelled":
                yield {"type": "done", "finish_reason": "cancelled", "steps": steps}
                return
            text = "".join(raw)
            prose, items = parse_tool_calls(text)
            if not items:
                yield {"type": "done", "finish_reason": "length" if finish == "length" else "stop",
                       "steps": steps}
                return

            valid = [c for c in items if isinstance(c, ToolCall)]
            content = prose
            bad_raw = [f"{OPEN}{c.raw}{CLOSE}" for c in items if isinstance(c, ParseError)]
            if bad_raw:                    # let the model see what it wrote wrong
                content = (prose + "\n\n" + "\n".join(bad_raw)).strip()
            assistant = {"role": "assistant", "content": content}
            if valid:
                assistant["tool_calls"] = [{"type": "function", "id": c.id,
                                            "function": {"name": c.name, "arguments": c.arguments}}
                                           for c in valid]
            msgs.append(assistant)

            for item in items:
                if cancel.is_set():
                    yield {"type": "done", "finish_reason": "cancelled", "steps": steps}
                    return
                if isinstance(item, ParseError):
                    name, args = INVALID_TOOL, {"raw": truncate(item.raw, 300)}
                    yield {"type": "tool_call", "id": item.id, "name": name, "arguments": args}
                    ok = False
                    if item.unclosed and finish == "length":
                        output = CUT_OFF_MESSAGE
                    else:
                        output = (f"Error: could not parse your tool call: {item.message}. Write exactly "
                                  f'{OPEN}{{"name": "<tool>", "arguments": {{...}}}}{CLOSE} with valid JSON.')
                else:
                    name, args = item.name, item.arguments
                    yield {"type": "tool_call", "id": item.id, "name": name, "arguments": args}
                    try:
                        ok, output = self.executor(name, args)
                    except Exception as err:            # executors should not raise; be safe anyway
                        ok, output = False, f"Error: {type(err).__name__}: {err}"
                    output = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
                self.calls.append({"id": item.id, "name": name, "arguments": args, "ok": ok, "output": output})
                tail = name in TAIL_TOOLS
                yield {"type": "tool_result", "id": item.id, "ok": ok,
                       "output": truncate(output, EVENT_OUTPUT_LIMIT, tail)}
                fed = truncate(output, MODEL_OUTPUT_LIMIT, tail)
                if not ok and not fed.lower().startswith("error"):
                    fed = "Error: " + fed
                msgs.append({"role": "tool", "name": name, "content": fed or "(no output)"})
            new_paragraph = bool("".join(self.prose).strip())
        yield {"type": "done", "finish_reason": "max_steps", "steps": steps}
