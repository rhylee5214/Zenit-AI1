"""Model registry: the single list of ZENIT checkpoints and what each one is.

models/registry.json holds one entry per exported model. Entries record
where the weights live, which run and dataset produced them, and their eval
results. Status values: experimental, candidate, released, retired.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .runlog import now_iso

# Development milestones. These describe goals, not achieved capability.
ROADMAP = [
    {"version": "0.1", "goal": "Pipeline proof: tiny native ZENIT trained from scratch on the CC0 seed corpus."},
    {"version": "1.0", "goal": "Initial experimental model: native ZENIT pretrained on licensed open text on a GPU."},
    {"version": "2.0", "goal": "Improved instruction following: SFT (LoRA/QLoRA) on an open-weight foundation model."},
    {"version": "3.0", "goal": "Improved conversation: multi-turn SFT data, persona tuning, memory-aware prompts."},
    {"version": "4.0", "goal": "Better coding and reasoning: code and math SFT, preference optimisation (DPO)."},
    {"version": "5.0", "goal": "Advanced model: larger base, curated mixture, retrieval-aware training, wider evals."},
    {"version": "5.5", "goal": "Major refinement: accuracy, hallucination control, memory and instruction following."},
]


class Registry:
    def __init__(self, models_dir: str | Path = "models") -> None:
        self.path = Path(models_dir) / "registry.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"models": [], "roadmap": ROADMAP}

    def _write(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def list(self) -> list[dict]:
        return self._read()["models"]

    def get(self, model_id: str) -> dict | None:
        return next((m for m in self.list() if m["id"] == model_id), None)

    def register(self, entry: dict) -> dict:
        data = self._read()
        entry = {"status": "experimental", "created": now_iso(), "evals": [], **entry}
        data["models"] = [m for m in data["models"] if m["id"] != entry["id"]] + [entry]
        self._write(data)
        return entry

    def update(self, model_id: str, **fields) -> dict:
        data = self._read()
        for m in data["models"]:
            if m["id"] == model_id:
                m.update(fields)
                self._write(data)
                return m
        raise KeyError(model_id)

    def add_eval(self, model_id: str, summary: dict) -> None:
        data = self._read()
        for m in data["models"]:
            if m["id"] == model_id:
                m.setdefault("evals", []).append(summary)
                self._write(data)
                return
        raise KeyError(model_id)
