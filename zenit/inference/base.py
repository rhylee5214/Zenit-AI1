"""Engine interface shared by every ZENIT backend."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class GenerationParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    seed: int | None = None
    stop: list[str] = field(default_factory=list)


@dataclass
class GenerationEvent:
    type: str                    # "token" | "done"
    text: str = ""
    finish_reason: str | None = None   # stop | length | cancelled | context
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Engine:
    """Subclasses implement load(), count_tokens(), stream() and info().

    supports_tools: the engine accepts a `tools` list in stream() and its
    chat template teaches the model to emit <tool_call> blocks.
    generation: per-model sampling defaults from configs/serve.json (see
    Persona.sampling for how they combine with behaviour profiles)."""

    id: str = "engine"
    context_window: int = 2048
    short_system_prompt: bool = False
    supports_tools: bool = False
    generation: dict = {}

    def load(self) -> None:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def count_messages(self, messages: list[dict]) -> int:
        return sum(self.count_tokens(m["content"]) + 4 for m in messages) + 2

    def stream(self, messages: list[dict], params: GenerationParams,
               cancel: threading.Event | None = None,
               tools: list[dict] | None = None) -> Iterator[GenerationEvent]:
        raise NotImplementedError

    def info(self) -> dict:
        return {"id": self.id, "context_window": self.context_window}

    def generate(self, messages: list[dict], params: GenerationParams) -> tuple[str, GenerationEvent]:
        parts, last = [], None
        for ev in self.stream(messages, params):
            if ev.type == "token":
                parts.append(ev.text)
            else:
                last = ev
        return "".join(parts), last
