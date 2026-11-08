"""Builds the set of servable models from configs/serve.json, falling back
to every model in models/registry.json.

Relative paths in serve.json ("path", "adapter") are resolved against the
server root when that local directory exists, so the server works from any
working directory. Anything else (for example a Hugging Face hub id such as
"Qwen/Qwen3-4B-Base") is passed through unchanged.

Per-model extras understood for every engine:
  card            model card facts used when no model_card.json sits next to
                  the weights (version, base_model, license, ...)
  supports_tools  the engine can take part in tool-calling loops (hf only)
  generation      sampling defaults for this model (temperature, top_p,
                  top_k, repetition_penalty, max_tokens); see
                  Persona.sampling for how they combine with profiles
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import Engine
from .hf_engine import HFEngine
from .native import NativeEngine


def resolve_path(value: str | None, root: Path | None) -> str | None:
    if not value or root is None:
        return value
    p = Path(value)
    if p.is_absolute():
        return value
    candidate = root / p
    return str(candidate) if candidate.exists() else value


class ModelPool:
    def __init__(self, serve_config: str | Path = "configs/serve.json",
                 models_dir: str | Path = "models", root: str | Path | None = None) -> None:
        self.engines: dict[str, Engine] = {}
        self.specs: dict[str, dict] = {}
        self.default: str | None = None
        self.root = Path(root).resolve() if root is not None else Path(models_dir).resolve().parent
        cfg_path = Path(serve_config)
        specs: list[dict] = []
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            specs = [dict(s) for s in cfg.get("models", []) if s.get("enabled", True)]
            self.default = cfg.get("default_model")
        reg = Path(models_dir) / "registry.json"
        if reg.exists():
            known = {s["id"] for s in specs}
            for m in json.loads(reg.read_text(encoding="utf-8")).get("models", []):
                path = resolve_path(m["path"], self.root)
                if m["id"] not in known and m.get("status") != "retired" and Path(path).exists():
                    specs.append({"id": m["id"], "engine": "native", "path": path})
        for s in specs:
            s["path"] = resolve_path(s.get("path"), self.root)
            if s.get("adapter"):
                s["adapter"] = resolve_path(s["adapter"], self.root)
            self.specs[s["id"]] = s
            if s.get("engine", "native") == "native":
                eng = NativeEngine(s["path"], s["id"], s.get("backend", "auto"), s.get("device", "auto"))
                eng.generation = dict(s.get("generation") or {})
                self.engines[s["id"]] = eng
            elif s["engine"] == "hf":
                self.engines[s["id"]] = HFEngine(s["id"], s["path"], s.get("adapter"),
                                                 s.get("load_in_4bit", False), s.get("device_map", "auto"),
                                                 s.get("chat_template_kwargs"), s.get("context_window", 8192),
                                                 card=s.get("card"), supports_tools=s.get("supports_tools", False),
                                                 generation=s.get("generation"))
            else:
                raise ValueError(f"unknown engine type {s['engine']}")
        if self.default not in self.engines:
            self.default = next(iter(self.engines), None)

    def get(self, model_id: str | None) -> Engine:
        mid = model_id or self.default
        if mid not in self.engines:
            raise KeyError(f"model not available: {mid}")
        eng = self.engines[mid]
        eng.load()
        return eng

    def peek(self, model_id: str | None) -> Engine | None:
        """The engine object without loading its weights (None if unknown)."""
        return self.engines.get(model_id or self.default)

    def list(self) -> list[dict]:
        out = []
        for mid, spec in self.specs.items():
            eng = self.engines[mid]
            info = {"id": mid, "name": spec.get("name") or mid, "engine": spec.get("engine", "native"), "default": mid == self.default,
                    "loaded": bool(getattr(eng, "loaded", False)),
                    "supports_tools": bool(getattr(eng, "supports_tools", False))}
            if getattr(eng, "loaded", False):
                info.update(eng.info())
            else:
                card_file = Path(spec.get("path") or "") / "model_card.json"
                c = None
                if spec.get("path") and card_file.exists():
                    c = json.loads(card_file.read_text(encoding="utf-8"))
                elif spec.get("card"):
                    c = spec["card"]
                if c:
                    info.update(version=c.get("version"), parameters=c.get("parameters"),
                                stage=c.get("stage"), base_model=c.get("base_model"))
            out.append(info)
        return out
